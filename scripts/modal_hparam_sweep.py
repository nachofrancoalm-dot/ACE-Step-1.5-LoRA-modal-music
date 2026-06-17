#!/usr/bin/env python3
"""Stage 1 hyperparameter sweep for modal LoRA: 24 experiments, results to CSV."""

import argparse
import csv
import errno
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

try:
    from loguru import logger

    HAS_LOGURU = True
except ModuleNotFoundError:
    logger = logging.getLogger("modal_hparam_sweep_stage1")
    HAS_LOGURU = False

STAGE1_GRID = {
    "frozen_mode": ["B"],  # Only LoRA (Mode B) for Stage 1
    "rank": [32, 64, 128],
    "alpha": [32, 128],
    "lr": [1e-4, 1e-3],
    "dropout": [0.05, 0.1],
}

FIXED_CONFIG = {
    "optimizer": "adamw",
    "scheduler": "cosine",
    "epochs": 140,
    "batch_size": 4,
    "save_every": 20,
    "model_variant": "turbo",
    "num_inference_steps": 8,
    "guidance_scale": 4.0,
}

_DISK_FULL_WARNING_EMITTED = False


def _is_no_space_left_error(exc: BaseException) -> bool:
    return isinstance(exc, OSError) and exc.errno == errno.ENOSPC


class _SafeFileHandler(logging.FileHandler):
    """File handler that self-disables on ENOSPC to avoid traceback spam."""

    def emit(self, record: logging.LogRecord) -> None:
        global _DISK_FULL_WARNING_EMITTED

        if getattr(self, "_disabled_due_to_disk_full", False):
            return

        try:
            super().emit(record)
        except OSError as exc:
            if not _is_no_space_left_error(exc):
                raise

            self._disabled_due_to_disk_full = True
            try:
                self.close()
            except Exception:
                pass

            if not _DISK_FULL_WARNING_EMITTED:
                _DISK_FULL_WARNING_EMITTED = True
                try:
                    sys.__stderr__.write(
                        "WARNING: Disk is full. File logging disabled for modal_hparam_sweep_stage1.\n"
                    )
                except Exception:
                    pass


def _has_min_free_space(path: Path, min_free_bytes: int = 256 * 1024 * 1024) -> bool:
    usage = shutil.disk_usage(path)
    return usage.free >= min_free_bytes


def setup_logging(output_dir: Path) -> None:
    log_file = output_dir / "sweep_stage1.log"
    if HAS_LOGURU:
        logger.remove()
        logger.add(
            sys.stderr,
            format="<level>{level: <8}</level> | <cyan>{name}</cyan> - {message}",
            level="INFO",
        )
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} - {message}",
            level="DEBUG",
        )
        return

    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter("%(levelname)-8s | %(name)s - %(message)s"))

    logger.addHandler(stream_handler)

    try:
        file_handler = _SafeFileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s - %(message)s")
        )
        logger.addHandler(file_handler)
    except OSError as exc:
        if _is_no_space_left_error(exc):
            logger.warning(
                "No space left while creating log file. Continuing with console logging only."
            )
        else:
            raise


def generate_grid(grid_config: dict[str, list]) -> list[dict[str, Any]]:
    keys = list(grid_config.keys())
    values = [grid_config[k] for k in keys]
    combinations = []
    for combo in product(*values):
        combinations.append(dict(zip(keys, combo)))
    return combinations


def summarize_error(output: str, max_lines: int = 8, max_chars: int = 800) -> str:
    if not output:
        return ""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    tail = lines[-max_lines:] if lines else []
    summary = " | ".join(tail)
    return summary[:max_chars]


def find_latest_checkpoint_dir(exp_output_dir: Path) -> Path | None:
    ckpt_root = exp_output_dir / "checkpoints"
    if not ckpt_root.exists() or not ckpt_root.is_dir():
        return None

    checkpoint_dirs = [p for p in ckpt_root.glob("epoch_*_loss_*") if p.is_dir()]
    if not checkpoint_dirs:
        return None

    def _epoch_num(path: Path) -> int:
        name = path.name
        if not name.startswith("epoch_"):
            return -1
        try:
            return int(name.split("_")[1])
        except Exception:
            return -1

    checkpoint_dirs.sort(key=lambda p: (_epoch_num(p), p.name))
    return checkpoint_dirs[-1]


def build_train_command(
    exp_id: int,
    params: dict[str, Any],
    dataset_dir: Path,
    output_base: Path,
    seed: int,
    resume_from: Path | None = None,
) -> list[str]:
    exp_name = f"exp_{exp_id:03d}_{params['frozen_mode']}_r{params['rank']}_a{params['alpha']}_lr{params['lr']:.0e}_do{params['dropout']}"
    output_dir = output_base / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "train.py",
        "--plain",
        "--yes",
        "fixed",
        "--checkpoint-dir",
        str(Path("./checkpoints")),
        "--dataset-dir",
        str(dataset_dir),
        "--output-dir",
        str(output_dir),
        "--model-variant",
        FIXED_CONFIG["model_variant"],
        "--rank",
        str(params["rank"]),
        "--alpha",
        str(params["alpha"]),
        "--lr",
        str(params["lr"]),
        "--dropout",
        str(params["dropout"]),
        "--epochs",
        str(FIXED_CONFIG["epochs"]),
        "--batch-size",
        str(FIXED_CONFIG["batch_size"]),
        "--save-every",
        str(FIXED_CONFIG["save_every"]),
        "--seed",
        str(seed),
        "--optimizer-type",
        FIXED_CONFIG["optimizer"],
        "--scheduler-type",
        FIXED_CONFIG["scheduler"],
    ]

    if resume_from is not None:
        cmd.extend(["--resume-from", str(resume_from)])

    # train.py fixed CLI does not expose --frozen-layer-type; frozen_mode=B is the default
    # trainer behavior (base frozen + LoRA trainable). Other modes are not implemented.
    if params.get("frozen_mode") != "B":
        logger.warning(
            f"frozen_mode={params.get('frozen_mode')} requested but this script currently supports only mode B with current train.py CLI"
        )

    return cmd, str(output_dir), exp_name


def run_training(
    cmd: list[str],
    timeout_hours: float = 2.0,
    verbose: bool = True,
    live_log_path: Path | None = None,
) -> tuple[int, str]:
    """Run a subprocess command with timeout; returns (exit_code, combined_output)."""
    timeout_sec = timeout_hours * 3600
    try:
        logger.info(f"Starting: {' '.join(cmd[:5])}...")
        if live_log_path is None:
            live_log_path = Path("train_stdout_stderr.log")

        live_log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Live log: {live_log_path}")

        start_ts = time.time()
        last_heartbeat = 0.0

        with open(live_log_path, "w", encoding="utf-8") as log_f:
            process = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=Path.cwd(),
                bufsize=1,
            )

            while True:
                return_code = process.poll()
                if return_code is not None:
                    break

                elapsed = time.time() - start_ts
                if elapsed - last_heartbeat >= 60:
                    logger.info(
                        f"Training still running ({elapsed/60:.1f} min elapsed)"
                    )
                    last_heartbeat = elapsed

                if elapsed > timeout_sec:
                    process.kill()
                    logger.error(f"Training timeout after {timeout_hours} hours")
                    return -1, "TIMEOUT"

                time.sleep(5)

        output = live_log_path.read_text(encoding="utf-8", errors="replace")
        return return_code, output
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return -1, str(e)


def run_evaluation(
    exp_output_dir: Path, epoch_checkpoint: int = 140, seed: int = 1111
) -> dict[str, float]:
    # Placeholder — not called in the main sweep loop; evaluation goes through evaluate_checkpoint().
    logger.info(f"Evaluating checkpoint from {exp_output_dir}")
    return {
        "auto_score_0_1": 0.0,
        "modal_accuracy": 0.0,
        "in_mode_ratio": 0.0,
        "tonal_stability": 0.0,
        "tonic_dominance": 0.0,
        "template_similarity": 0.0,
    }


def evaluate_checkpoint(
    exp_output_dir: str,
    prompts_block: str = "E",
    seeds: tuple[int, int] = (1111, 2222),
) -> dict[str, float]:
    checkpoint_root = Path(exp_output_dir)
    infer_manifest = checkpoint_root / "modal_batch_manifest.csv"
    copied_audio_dir = checkpoint_root / "generated_audio"
    eval_csv = checkpoint_root / "modal_eval_metrics.csv"

    cmd_infer = [
        "python",
        "scripts/modal_batch_infer.py",
        "--checkpoints-root",
        str(checkpoint_root),
        "--prompts-block",
        prompts_block,
        "--seeds",
        *[str(seed) for seed in seeds],
        "--output-csv",
        str(infer_manifest),
        "--copy-audio-dir",
        str(copied_audio_dir),
    ]

    logger.info(f"Running inference for {exp_output_dir}...")
    exit_code, output = run_training(cmd_infer, timeout_hours=1.0)

    if exit_code != 0:
        logger.error(f"Inference failed: {output}")
        return {
            "auto_score_0_1": -1.0,
            "modal_accuracy": -1.0,
            "in_mode_ratio": -1.0,
            "tonal_stability": -1.0,
            "tonic_dominance": -1.0,
            "template_similarity": -1.0,
        }

    cmd_eval = [
        "python",
        "scripts/modal_eval_pipeline.py",
        "--input-dir",
        str(copied_audio_dir),
        "--output-csv",
        str(eval_csv),
        "--recursive",
    ]

    logger.info("Running evaluation pipeline...")
    exit_code, output = run_training(cmd_eval, timeout_hours=0.5)
    if exit_code != 0:
        logger.error(f"Evaluation failed: {output}")
        return {
            "auto_score_0_1": -1.0,
            "modal_accuracy": -1.0,
            "in_mode_ratio": -1.0,
            "tonal_stability": -1.0,
            "tonic_dominance": -1.0,
            "template_similarity": -1.0,
        }

    if not eval_csv.exists():
        logger.error(f"Evaluation CSV not found at {eval_csv}")
        return {
            "auto_score_0_1": -1.0,
            "modal_accuracy": -1.0,
            "in_mode_ratio": -1.0,
            "tonal_stability": -1.0,
            "tonic_dominance": -1.0,
            "template_similarity": -1.0,
        }

    rows: list[dict[str, str]] = []
    with open(eval_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "ok" and row.get("mode_inferred") != "unknown":
                rows.append(row)

    if not rows:
        logger.error("Evaluation produced no valid rows")
        return {
            "auto_score_0_1": -1.0,
            "modal_accuracy": -1.0,
            "in_mode_ratio": -1.0,
            "tonal_stability": -1.0,
            "tonic_dominance": -1.0,
            "template_similarity": -1.0,
        }

    def _mean(key: str) -> float:
        values: list[float] = []
        for row in rows:
            value = row.get(key, "")
            if not value:
                continue
            try:
                values.append(float(value))
            except ValueError:
                continue
        return sum(values) / len(values) if values else -1.0

    # frame_in_mode_ratio is the closest proxy to modal accuracy in this pipeline.
    modal_accuracy = _mean("frame_in_mode_ratio")

    return {
        "auto_score_0_1": _mean("auto_score_0_1"),
        "modal_accuracy": modal_accuracy,
        "in_mode_ratio": _mean("in_mode_ratio"),
        "tonal_stability": _mean("tonal_stability"),
        "tonic_dominance": _mean("tonic_dominance_ratio"),
        "template_similarity": _mean("template_similarity"),
    }


def sweep_stage1(
    dataset_dir: Path,
    output_dir: Path,
    seed: int = 1111,
    parallel: int = 1,
    evaluate: bool = True,
    resume: bool = False,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir)

    logger.info("=" * 80)
    logger.info(f"STAGE 1 HYPERPARAMETER SWEEP START")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Dataset: {dataset_dir}")
    logger.info(f"Grid: {json.dumps(STAGE1_GRID, default=str, indent=2)}")
    logger.info(f"Fixed config: {json.dumps(FIXED_CONFIG, indent=2)}")
    logger.info("=" * 80)

    combinations = generate_grid(STAGE1_GRID)
    total = len(combinations)

    logger.info(f"Generated {total} experiment combinations")

    results_file = output_dir / "stage1_results.csv"
    fieldnames = [
        "exp_id",
        "frozen_mode",
        "rank",
        "alpha",
        "lr",
        "dropout",
        "epoch_checkpoint",
        "auto_score_0_1",
        "modal_accuracy",
        "in_mode_ratio",
        "tonal_stability",
        "tonic_dominance",
        "template_similarity",
        "training_status",
        "training_duration_sec",
        "training_error",
    ]

    completed_ids: set[int] = set()
    if resume and results_file.exists():
        try:
            with open(results_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    exp_id_raw = row.get("exp_id", "")
                    try:
                        completed_ids.add(int(exp_id_raw))
                    except Exception:
                        continue
            logger.info(
                f"Resume mode: found {len(completed_ids)} logged experiments in {results_file}"
            )
        except Exception as e:
            logger.warning(f"Could not load existing results CSV for resume: {e}")
    else:
        with open(results_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    logger.info(f"Results will be saved to: {results_file}")

    start_time_global = time.time()
    success_count = 0
    fail_count = 0

    skipped_count = 0

    for exp_id, params in enumerate(combinations, start=1):
        if resume and exp_id in completed_ids:
            skipped_count += 1
            continue

        if skipped_count > 0:
            logger.info(
                f"Resume mode: skipped {skipped_count} already-logged experiments"
            )
            skipped_count = 0

        if not _has_min_free_space(output_dir):
            logger.error(
                "Insufficient free disk space (<256 MB). "
                "Stopping sweep before launching next experiment."
            )
            fail_count += 1
            break

        logger.info(f"\n{'='*80}")
        logger.info(f"[{exp_id}/{total}] Experiment: {params}")
        logger.info(f"{'='*80}")

        exp_name = f"exp_{exp_id:03d}_{params['frozen_mode']}_r{params['rank']}_a{params['alpha']}_lr{params['lr']:.0e}_do{params['dropout']}"
        exp_output_dir_path = output_dir / exp_name
        resume_from_ckpt = None
        if resume:
            resume_from_ckpt = find_latest_checkpoint_dir(exp_output_dir_path)
            if resume_from_ckpt is not None:
                logger.info(
                    f"Resume mode: will resume exp {exp_id} from checkpoint {resume_from_ckpt}"
                )

        try:
            cmd, exp_output_dir, exp_name = build_train_command(
                exp_id,
                params,
                dataset_dir,
                output_dir,
                seed,
                resume_from=resume_from_ckpt,
            )
        except OSError as exc:
            if _is_no_space_left_error(exc):
                logger.error(
                    f"No space left while creating output directory for exp {exp_id}. "
                    "Stopping sweep so it can be resumed after freeing space."
                )
                fail_count += 1
                break
            raise

        start_time_exp = time.time()
        exp_log_file = Path(exp_output_dir) / "train_stdout_stderr.log"
        exit_code, training_output = run_training(
            cmd,
            timeout_hours=2.0,
            live_log_path=exp_log_file,
        )
        duration_sec = time.time() - start_time_exp

        training_status = "SUCCESS" if exit_code == 0 else "FAILED"
        training_error = "" if exit_code == 0 else summarize_error(training_output)

        logger.info(
            f"Training {training_status} | Duration: {duration_sec:.1f}s | Exit code: {exit_code}"
        )

        if exit_code == 0:
            success_count += 1
        else:
            fail_count += 1

        eval_metrics = {
            "auto_score_0_1": -1.0,
            "modal_accuracy": -1.0,
            "in_mode_ratio": -1.0,
            "tonal_stability": -1.0,
            "tonic_dominance": -1.0,
            "template_similarity": -1.0,
        }

        if exit_code == 0 and evaluate:
            try:
                eval_metrics = evaluate_checkpoint(exp_output_dir)
                logger.info(f"Evaluation metrics: {eval_metrics}")
            except Exception as e:
                logger.error(f"Evaluation failed: {e}")

        row = {
            "exp_id": exp_id,
            "frozen_mode": params["frozen_mode"],
            "rank": params["rank"],
            "alpha": params["alpha"],
            "lr": f"{params['lr']:.0e}",
            "dropout": params["dropout"],
            "epoch_checkpoint": FIXED_CONFIG["epochs"],
            "auto_score_0_1": eval_metrics.get("auto_score_0_1", -1.0),
            "modal_accuracy": eval_metrics.get("modal_accuracy", -1.0),
            "in_mode_ratio": eval_metrics.get("in_mode_ratio", -1.0),
            "tonal_stability": eval_metrics.get("tonal_stability", -1.0),
            "tonic_dominance": eval_metrics.get("tonic_dominance", -1.0),
            "template_similarity": eval_metrics.get("template_similarity", -1.0),
            "training_status": training_status,
            "training_duration_sec": f"{duration_sec:.1f}",
            "training_error": training_error,
        }

        with open(results_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

        logger.info(f"Results saved to: {results_file}")

    total_duration = time.time() - start_time_global
    logger.info(f"\n{'='*80}")
    logger.info(f"STAGE 1 SWEEP COMPLETE")
    logger.info(f"Total experiments: {total}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {fail_count}")
    logger.info(f"Total duration: {total_duration/3600:.1f} hours")
    logger.info(f"Results: {results_file}")
    logger.info(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 Hyperparameter Sweep for Modal LoRA"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to preprocessed modal dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Base output directory for all experiments",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1111,
        help="Random seed for reproducibility (default: 1111)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel jobs (currently not implemented; must be 1)",
    )
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Skip evaluation step (only run training)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output-dir: skip logged exp_ids and auto-resume from latest checkpoint when present",
    )

    args = parser.parse_args()

    if args.parallel != 1:
        logger.warning("Parallel execution not yet implemented; forcing serial (--parallel 1)")

    sweep_stage1(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        parallel=args.parallel,
        evaluate=not args.no_evaluate,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
