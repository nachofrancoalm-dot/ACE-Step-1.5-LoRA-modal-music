#!/usr/bin/env python3
"""Analyze generated audio metrics and rank best checkpoints per experiment.

Groups generated audios by checkpoint (epoch) and shows top checkpoints for each experiment,
allowing quick selection of what to listen to manually.

Usage:
    python scripts/rank_checkpoints_per_experiment.py \
        --results-csv ./hparam_sweep_stage1_results/stage1_results.csv \
        --sweep-output-dir ./hparam_sweep_stage1_results \
        --top-n 5
"""

import argparse
import csv
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


DEBUG = False


def _debug(msg: str) -> None:
    if DEBUG:
        print(f"[DEBUG] {msg}")


def _parse_score_0_1(row: Dict[str, str]) -> float:
    value_01 = (row.get("auto_score_0_1") or "").strip()
    if value_01:
        try:
            parsed_01 = float(value_01)
            if parsed_01 >= 0:
                return parsed_01
        except ValueError:
            pass

    value_15 = (row.get("auto_score_1_5") or "").strip()
    if value_15:
        try:
            parsed_15 = float(value_15)
            return max(0.0, min(1.0, (parsed_15 - 1.0) / 4.0))  # legacy [1,5] → [0,1]
        except ValueError:
            pass

    return -1.0


def _extract_epoch(filename: str) -> int:
    match = re.search(r"epoch_(\d+)", filename)
    if match:
        return int(match.group(1))
    return -1


def _resolve_sweep_dir(requested_dir: Path) -> Path:
    _debug(f"Requested sweep dir: {requested_dir}")
    if requested_dir.exists():
        _debug("Requested sweep dir exists; using it directly")
        return requested_dir

    fallback_names = []
    if requested_dir.name == "hparam_sweep_stage1_results":
        fallback_names.append("hparam_sweep")
    elif requested_dir.name == "hparam_sweep":
        fallback_names.append("hparam_sweep_stage1_results")

    for fallback_name in fallback_names:
        fallback_dir = requested_dir.parent / fallback_name
        _debug(f"Trying fallback sweep dir: {fallback_dir}")
        if fallback_dir.exists():
            print(f"[INFO] Using fallback sweep directory: {fallback_dir}")
            return fallback_dir

    return requested_dir


def _find_exp_dir(sweep_dir: Path, exp_id: str, row: Dict[str, str]) -> Path | None:
    frozen_mode = row.get("frozen_mode", "B")
    rank = row.get("rank", "")
    alpha = row.get("alpha", "")
    lr = row.get("lr", "")
    dropout = row.get("dropout", "")

    candidates = [
        sweep_dir / f"exp_{int(exp_id):03d}_{frozen_mode}_r{rank}_a{alpha}_lr{lr}_do{dropout}",
        sweep_dir / f"exp_{int(exp_id)}_{frozen_mode}_r{rank}_a{alpha}_lr{lr}_do{dropout}",
    ]

    for candidate in candidates:
        _debug(f"Checking candidate exp dir for exp_{exp_id}: {candidate}")
        if candidate.exists():
            _debug(f"Matched exact candidate for exp_{exp_id}: {candidate}")
            return candidate

    matches = sorted(sweep_dir.glob(f"exp_{int(exp_id):03d}_*"))
    _debug(f"Glob exp_{int(exp_id):03d}_* matches: {[m.name for m in matches]}")
    if matches:
        return matches[0]

    matches = sorted(sweep_dir.glob(f"exp_{int(exp_id)}_*"))
    _debug(f"Glob exp_{int(exp_id)}_* matches: {[m.name for m in matches]}")
    if matches:
        return matches[0]

    return None


def _extract_exp_metadata(exp_dir: Path) -> Dict[str, str]:
    name = exp_dir.name
    match = re.search(
        r"exp_(?P<exp_id>\d+)_\w+_r(?P<rank>[^_]+)_a(?P<alpha>[^_]+)_lr(?P<lr>[^_]+)_do(?P<dropout>[^_]+)",
        name,
    )
    if not match:
        return {}

    return {
        "exp_id": match.group("exp_id"),
        "rank": match.group("rank"),
        "alpha": match.group("alpha"),
        "lr": match.group("lr"),
        "dropout": match.group("dropout"),
    }


def _ensure_eval_metrics(exp_dir: Path) -> Path | None:
    metrics_csv = exp_dir / "modal_eval_metrics.csv"
    _debug(f"Looking for eval CSV: {metrics_csv}")
    if metrics_csv.exists() and metrics_csv.stat().st_size > 0:
        _debug(f"Found eval CSV ({metrics_csv.stat().st_size} bytes)")
        return metrics_csv

    generated_audio = exp_dir / "generated_audio"
    _debug(f"Eval CSV missing/empty; checking generated audio dir: {generated_audio}")
    if not generated_audio.exists():
        _debug("generated_audio folder does not exist")
        return None

    if not any(generated_audio.rglob("*.flac")):
        _debug("No .flac files found under generated_audio")
        return None

    _debug("Found audio files; running modal_eval_pipeline.py to generate metrics")

    cmd = [
        "python",
        "scripts/modal_eval_pipeline.py",
        "--input-dir",
        str(generated_audio),
        "--output-csv",
        str(metrics_csv),
        "--recursive",
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(Path.cwd()),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except Exception as exc:
        print(f"[WARN] Could not run evaluation for {exp_dir}: {exc}")
        return None

    if result.returncode != 0 or not metrics_csv.exists():
        tail = result.stdout[-800:] if result.stdout else ""
        print(f"[WARN] Evaluation failed for {exp_dir}\n{tail}")
        return None

    _debug(f"Generated eval CSV at {metrics_csv}")

    return metrics_csv


def _load_exp_metrics(exp_dir: Path) -> Dict[int, List[Dict]]:
    metrics_csv = _ensure_eval_metrics(exp_dir)
    if metrics_csv is None:
        return {}

    by_epoch: Dict[int, List[Dict]] = defaultdict(list)
    total_rows = 0
    skipped_status = 0
    skipped_epoch = 0
    skipped_score = 0
    first_headers: List[str] = []
    
    try:
        with metrics_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader:
                return {}
            first_headers = list(reader.fieldnames or [])
            _debug(f"CSV headers for {metrics_csv}: {first_headers}")
            has_status_col = "status" in first_headers
            for row in reader:
                total_rows += 1
                status = (row.get("status") or "").strip().lower()
                if has_status_col and status and status != "ok":
                    skipped_status += 1
                    continue

                filename = row.get("file", row.get("path", ""))
                epoch = _extract_epoch(filename)
                if epoch < 0:
                    skipped_epoch += 1
                    continue

                try:
                    score = _parse_score_0_1(row)
                    if score >= 0:
                        by_epoch[epoch].append({
                            "file": filename,
                            "mode": row.get("mode_inferred", "unknown"),
                            "auto_score_0_1": score,
                            "in_mode_ratio": float(row.get("in_mode_ratio", "-1")),
                            "tonal_stability": float(row.get("tonal_stability", "-1")),
                            "template_similarity": float(row.get("template_similarity", "-1")),
                        })
                    else:
                        skipped_score += 1
                except Exception:
                    skipped_score += 1
                    continue
    except Exception as e:
        print(f"[WARN] Could not read {metrics_csv}: {e}")
        return {}

    _debug(
        "Rows summary "
        f"for {metrics_csv}: total={total_rows}, "
        f"kept={sum(len(v) for v in by_epoch.values())}, "
        f"skipped_status={skipped_status}, skipped_epoch={skipped_epoch}, skipped_score={skipped_score}"
    )
    _debug(f"Epoch keys parsed: {sorted(by_epoch.keys())}")
    
    return by_epoch


def _aggregate_by_epoch(by_epoch: Dict[int, List[Dict]]) -> Dict[int, Dict[str, float]]:
    aggregated = {}
    for epoch, rows in by_epoch.items():
        if not rows:
            continue
        
        n = len(rows)
        aggregated[epoch] = {
            "count": n,
            "auto_score_0_1": sum(r["auto_score_0_1"] for r in rows) / n,
            "in_mode_ratio": sum(r["in_mode_ratio"] for r in rows) / n,
            "tonal_stability": sum(r["tonal_stability"] for r in rows) / n,
            "template_similarity": sum(r["template_similarity"] for r in rows) / n,
        }
    return aggregated


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rank best checkpoints per experiment for manual evaluation"
    )
    parser.add_argument("--results-csv", type=Path, help="stage1_results.csv")
    parser.add_argument("--sweep-output-dir", type=Path, required=True, help="Sweep output dir")
    parser.add_argument(
        "--exp-ids",
        type=int,
        nargs="+",
        help="Experiment IDs to rank directly when stage1_results.csv is unavailable.",
    )
    parser.add_argument("--top-n", type=int, default=5, help="Number of top experiments")
    parser.add_argument("--top-ckpt-per-exp", type=int, default=3, help="Top checkpoints per exp")
    parser.add_argument("--debug", action="store_true", help="Print detailed path/debug traces")
    
    args = parser.parse_args()

    global DEBUG
    DEBUG = args.debug
    
    sweep_dir = _resolve_sweep_dir(args.sweep_output_dir.expanduser().resolve())
    _debug(f"Resolved sweep dir: {sweep_dir}")

    top_rows: List[Tuple[Dict[str, str], float]] = []
    if args.exp_ids:
        for exp_id in args.exp_ids:
            row: Dict[str, str] = {"exp_id": str(exp_id), "frozen_mode": "B"}
            exp_dir = _find_exp_dir(sweep_dir, str(exp_id), row)
            if exp_dir is None:
                print(f"[WARN] exp_{exp_id}: folder not found")
                continue
            _debug(f"exp_{exp_id} resolved dir: {exp_dir}")

            metrics_csv = exp_dir / "modal_eval_metrics.csv"
            score = -1.0
            if metrics_csv.exists():
                try:
                    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
                        reader = csv.DictReader(f)
                        scores = []
                        for metric_row in reader:
                            if metric_row.get("status") != "ok":
                                continue
                            parsed_score = _parse_score_0_1(metric_row)
                            if parsed_score >= 0:
                                scores.append(parsed_score)
                        if scores:
                            score = sum(scores) / len(scores)
                except Exception:
                    score = -1.0

            top_rows.append((row, score))

        if not top_rows:
            print("[FAIL] No experiment folders found for the provided exp ids")
            return 1
    else:
        if args.results_csv is None:
            print("[FAIL] Provide either --results-csv or --exp-ids")
            return 1

        csv_path = args.results_csv.expanduser().resolve()
        if not csv_path.exists():
            print(f"[FAIL] CSV not found: {csv_path}")
            return 1

        # Load and rank experiments from the global results CSV.
        rows = []
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("training_status", "").upper() == "SUCCESS":
                    try:
                        score = _parse_score_0_1(row)
                        if score >= 0:
                            rows.append((row, score))
                    except Exception:
                        pass

        rows.sort(key=lambda x: -x[1])
        top_rows = rows[:args.top_n]

        if not top_rows:
            print("[FAIL] No valid experiments found")
            return 1

    print("\n" + "=" * 100)
    print("CHECKPOINT RANKING FOR MANUAL EVALUATION")
    print("=" * 100)

    for exp_idx, (row, exp_score) in enumerate(top_rows, start=1):
        exp_id = row["exp_id"]
        exp_dir = _find_exp_dir(sweep_dir, exp_id, row)

        if exp_dir is None:
            print(f"\n[{exp_idx}] exp_{exp_id}: FOLDER NOT FOUND")
            continue

        by_epoch = _load_exp_metrics(exp_dir)
        if not by_epoch:
            print(f"\n[{exp_idx}] exp_{exp_id}: NO EVAL METRICS FOUND")
            continue

        metadata = dict(row)
        metadata.update(_extract_exp_metadata(exp_dir))
        rank = metadata.get("rank", "?")
        alpha = metadata.get("alpha", "?")
        lr = metadata.get("lr", "?")
        dropout = metadata.get("dropout", "?")
        
        aggregated = _aggregate_by_epoch(by_epoch)
        sorted_epochs = sorted(
            aggregated.items(),
            key=lambda x: -x[1]["auto_score_0_1"]
        )
        
        print(f"\n[{exp_idx}] exp_{exp_id} (config score={exp_score:.4f})")
        print(f"     r={rank}, alpha={alpha}, lr={lr}, dropout={dropout}")
        print(f"     {len(by_epoch)} checkpoints available")
        print()
        
        for rank_idx, (epoch, metrics) in enumerate(sorted_epochs[:args.top_ckpt_per_exp], start=1):
            print(f"     #{rank_idx} epoch_{epoch:03d}:")
            print(f"         auto_score    = {metrics['auto_score_0_1']:.4f}")
            print(f"         in_mode_ratio = {metrics['in_mode_ratio']:.4f}")
            print(f"         tonal_stab    = {metrics['tonal_stability']:.4f}")
            print(f"         template_sim  = {metrics['template_similarity']:.4f}")
            print(f"         samples       = {metrics['count']} (7 modes × 2 seeds)")
            print()
        
        print(f"     Folder: {exp_dir / 'generated_audio'}")
    
    print("=" * 100)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
