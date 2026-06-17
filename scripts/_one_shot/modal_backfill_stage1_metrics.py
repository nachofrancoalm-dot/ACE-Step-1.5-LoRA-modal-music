#!/usr/bin/env python3
"""Backfill missing Stage-1 modal metrics without retraining.

This script reads `stage1_results.csv`, finds rows with successful training but
missing metrics (e.g. auto_score_0_1 == -1.0), reruns only:
1) modal_batch_infer.py
2) modal_eval_pipeline.py

Then it writes aggregated metrics back to the same CSV (with backup).
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List


def _to_float(value: str, default: float = -1.0) -> float:
    """Parse float with fallback."""
    try:
        return float(value)
    except Exception:
        return default


def _mean(rows: List[Dict[str, str]], key: str) -> float:
    """Compute mean float for key, ignoring invalid values."""
    values: List[float] = []
    for row in rows:
        raw = row.get(key, "")
        if not raw:
            continue
        try:
            values.append(float(raw))
        except ValueError:
            continue
    if not values:
        return -1.0
    return sum(values) / len(values)


def _run(cmd: List[str], cwd: Path) -> tuple[int, str]:
    """Run command and return code + combined output."""
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode, proc.stdout


def _build_exp_dir(output_dir: Path, row: Dict[str, str]) -> Path:
    """Build experiment directory path from CSV row fields."""
    exp_id = int(row["exp_id"])
    frozen_mode = row["frozen_mode"]
    rank = int(float(row["rank"]))
    alpha = int(float(row["alpha"]))
    lr = float(row["lr"])
    dropout = float(row["dropout"])

    exp_name = (
        f"exp_{exp_id:03d}_{frozen_mode}_r{rank}_a{alpha}_"
        f"lr{lr:.0e}_do{dropout}"
    )
    return output_dir / exp_name


def _load_eval_metrics(eval_csv: Path) -> Dict[str, float]:
    """Aggregate evaluation CSV into stage1 metric fields."""
    valid_rows: List[Dict[str, str]] = []
    with eval_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status") == "ok" and row.get("mode_inferred") != "unknown":
                valid_rows.append(row)

    if not valid_rows:
        return {
            "auto_score_0_1": -1.0,
            "modal_accuracy": -1.0,
            "in_mode_ratio": -1.0,
            "tonal_stability": -1.0,
            "tonic_dominance": -1.0,
            "template_similarity": -1.0,
        }

    return {
        "auto_score_0_1": _mean(valid_rows, "auto_score_0_1"),
        "modal_accuracy": _mean(valid_rows, "frame_in_mode_ratio"),
        "in_mode_ratio": _mean(valid_rows, "in_mode_ratio"),
        "tonal_stability": _mean(valid_rows, "tonal_stability"),
        "tonic_dominance": _mean(valid_rows, "tonic_dominance_ratio"),
        "template_similarity": _mean(valid_rows, "template_similarity"),
    }


def _is_missing_metrics(row: Dict[str, str]) -> bool:
    """True when row looks unevaluated."""
    return _to_float(row.get("auto_score_0_1", "-1"), -1.0) < 0.0


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Backfill stage1_results.csv metrics from existing checkpoints"
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        required=True,
        help="Path to stage1_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root sweep output dir containing exp_* folders",
    )
    parser.add_argument(
        "--api-base-url",
        type=str,
        default="http://127.0.0.1:7860",
        help="ACE-Step API base URL for modal_batch_infer.py",
    )
    parser.add_argument(
        "--prompts-block",
        type=str,
        default="E",
        choices=["A", "B", "D", "E"],
        help="Prompt block used for evaluation",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1111, 2222],
        help="Seeds used by modal_batch_infer.py",
    )
    parser.add_argument(
        "--only-exp-ids",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of exp_id values to process",
    )
    parser.add_argument(
        "--cleanup-generated-audio",
        action="store_true",
        help="Delete exp_*/generated_audio after eval to save disk",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if one experiment fails",
    )

    args = parser.parse_args()

    results_csv = args.results_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not results_csv.exists():
        print(f"[FAIL] CSV not found: {results_csv}")
        return 1
    if not output_dir.exists():
        print(f"[FAIL] Output dir not found: {output_dir}")
        return 1

    rows: List[Dict[str, str]] = []
    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            print("[FAIL] CSV has no header")
            return 1
        for row in reader:
            rows.append(row)

    backup_csv = results_csv.with_suffix(results_csv.suffix + ".bak")
    shutil.copy2(results_csv, backup_csv)
    print(f"[INFO] Backup created: {backup_csv}")

    processed = 0
    updated = 0

    for row in rows:
        exp_id = int(row["exp_id"])
        training_status = row.get("training_status", "").upper()

        if training_status != "SUCCESS":
            continue
        if args.only_exp_ids and exp_id not in args.only_exp_ids:
            continue
        if not _is_missing_metrics(row):
            continue

        processed += 1
        exp_dir = _build_exp_dir(output_dir, row)
        if not exp_dir.exists():
            msg = f"[WARN] exp {exp_id}: folder not found: {exp_dir}"
            print(msg)
            if args.stop_on_error:
                return 1
            continue

        infer_manifest = exp_dir / "modal_batch_manifest.csv"
        generated_audio = exp_dir / "generated_audio"
        eval_csv = exp_dir / "modal_eval_metrics.csv"

        print(f"[INFO] exp {exp_id}: running inference...")
        infer_cmd = [
            "python",
            "scripts/modal_batch_infer.py",
            "--api-base-url",
            args.api_base_url,
            "--checkpoints-root",
            str(exp_dir),
            "--prompts-block",
            args.prompts_block,
            "--seeds",
            *[str(seed) for seed in args.seeds],
            "--output-csv",
            str(infer_manifest),
            "--copy-audio-dir",
            str(generated_audio),
        ]
        code, output = _run(infer_cmd, cwd=Path.cwd())
        if code != 0:
            print(f"[WARN] exp {exp_id}: inference failed")
            print(output[-1000:])
            if args.stop_on_error:
                return 1
            continue

        print(f"[INFO] exp {exp_id}: running eval pipeline...")
        eval_cmd = [
            "python",
            "scripts/modal_eval_pipeline.py",
            "--input-dir",
            str(generated_audio),
            "--output-csv",
            str(eval_csv),
            "--recursive",
        ]
        code, output = _run(eval_cmd, cwd=Path.cwd())
        if code != 0:
            print(f"[WARN] exp {exp_id}: eval failed")
            print(output[-1000:])
            if args.stop_on_error:
                return 1
            continue

        if not eval_csv.exists():
            print(f"[WARN] exp {exp_id}: eval CSV not found")
            if args.stop_on_error:
                return 1
            continue

        metrics = _load_eval_metrics(eval_csv)
        row["auto_score_0_1"] = f"{metrics['auto_score_0_1']:.6f}"
        row["modal_accuracy"] = f"{metrics['modal_accuracy']:.6f}"
        row["in_mode_ratio"] = f"{metrics['in_mode_ratio']:.6f}"
        row["tonal_stability"] = f"{metrics['tonal_stability']:.6f}"
        row["tonic_dominance"] = f"{metrics['tonic_dominance']:.6f}"
        row["template_similarity"] = f"{metrics['template_similarity']:.6f}"

        if args.cleanup_generated_audio and generated_audio.exists():
            shutil.rmtree(generated_audio, ignore_errors=True)

        updated += 1
        print(f"[OK] exp {exp_id}: metrics updated")

    with results_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n[SUMMARY]")
    print(f"Processed missing rows: {processed}")
    print(f"Updated rows: {updated}")
    print(f"CSV written: {results_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
