#!/usr/bin/env python3
"""Generate manual eval shortlist: collect top checkpoint audios for listening.

Copies audios from best checkpoints (per experiment) into an organized folder structure
for streamlined manual evaluation using the modal rubric.

Usage:
    python scripts/prepare_manual_eval_shortlist.py \
        --results-csv ./hparam_sweep_stage1_results/stage1_results.csv \
        --sweep-output-dir ./hparam_sweep_stage1_results \
        --output-dir ./manual_eval_shortlist \
        --top-n 5 \
        --top-ckpt-per-exp 2
"""

import argparse
import csv
import re
import subprocess
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def _extract_epoch(filename: str) -> int:
    match = re.search(r"epoch_(\d+)", filename)
    if match:
        return int(match.group(1))
    return -1


def _resolve_sweep_dir(requested_dir: Path) -> Path:
    if requested_dir.exists():
        return requested_dir

    fallback_names = []
    if requested_dir.name == "hparam_sweep_stage1_results":
        fallback_names.append("hparam_sweep")
    elif requested_dir.name == "hparam_sweep":
        fallback_names.append("hparam_sweep_stage1_results")

    for fallback_name in fallback_names:
        fallback_dir = requested_dir.parent / fallback_name
        if fallback_dir.exists():
            print(f"[INFO] Using fallback sweep directory: {fallback_dir}")
            return fallback_dir

    return requested_dir


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
            return max(0.0, min(1.0, (parsed_15 - 1.0) / 4.0))
        except ValueError:
            pass

    return -1.0


def _find_exp_dir(sweep_dir: Path, row: Dict[str, str]) -> Path | None:
    exp_id = int(row["exp_id"])
    frozen_mode = row.get("frozen_mode", "B")
    rank = row.get("rank", "")
    alpha = row.get("alpha", "")
    lr = row.get("lr", "")
    dropout = row.get("dropout", "")

    candidates = [
        sweep_dir / f"exp_{exp_id:03d}_{frozen_mode}_r{rank}_a{alpha}_lr{lr}_do{dropout}",
        sweep_dir / f"exp_{exp_id}_{frozen_mode}_r{rank}_a{alpha}_lr{lr}_do{dropout}",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(sweep_dir.glob(f"exp_{exp_id:03d}_*"))
    if matches:
        return matches[0]

    matches = sorted(sweep_dir.glob(f"exp_{exp_id}_*"))
    if matches:
        return matches[0]

    return None


def _ensure_eval_metrics(exp_dir: Path) -> Path | None:
    metrics_csv = exp_dir / "modal_eval_metrics.csv"
    if metrics_csv.exists() and metrics_csv.stat().st_size > 0:
        return metrics_csv

    generated_audio = exp_dir / "generated_audio"
    if not generated_audio.exists() or not any(generated_audio.rglob("*.flac")):
        return None

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
    except Exception:
        return None

    if result.returncode != 0 or not metrics_csv.exists():
        return None

    return metrics_csv


def _load_exp_metrics(exp_dir: Path) -> Dict[int, Dict]:
    metrics_csv = _ensure_eval_metrics(exp_dir)
    if metrics_csv is None:
        return {}
    
    by_epoch: Dict[int, List[Dict]] = defaultdict(list)
    
    try:
        with metrics_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                status = (row.get("status") or "").strip().lower()
                if status and status != "ok":
                    continue

                file_col = row.get("file", row.get("path", ""))
                epoch = _extract_epoch(file_col)
                if epoch < 0:
                    continue

                auto_score_val = _parse_score_0_1(row)
                if auto_score_val < 0:
                    continue

                by_epoch[epoch].append({
                    "file": file_col,
                    "score": auto_score_val,
                })
    except Exception as e:
        print(f"[WARN] Could not read {metrics_csv}: {e}")
        return {}
    
    aggregated = {}
    for epoch, rows in by_epoch.items():
        if rows:
            aggregated[epoch] = {
                "score": sum(r["score"] for r in rows) / len(rows),
                "count": len(rows),
            }
    
    return aggregated


def _load_experiment_score(exp_dir: Path) -> float:
    metrics_csv = exp_dir / "modal_eval_metrics.csv"
    if not metrics_csv.exists():
        return -1.0

    scores: List[float] = []
    try:
        with metrics_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                status = (row.get("status") or "").strip().lower()
                if status and status != "ok":
                    continue
                score = _parse_score_0_1(row)
                if score >= 0:
                    scores.append(score)
    except Exception:
        return -1.0

    if not scores:
        return -1.0
    return sum(scores) / len(scores)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare manual eval shortlist")
    parser.add_argument("--results-csv", type=Path)
    parser.add_argument("--sweep-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("manual_eval_shortlist"))
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--top-ckpt-per-exp", type=int, default=2)
    parser.add_argument(
        "--exp-ids",
        type=int,
        nargs="+",
        help="Experiment IDs to shortlist directly when the global results CSV is unavailable.",
    )
    
    args = parser.parse_args()
    
    sweep_dir = _resolve_sweep_dir(args.sweep_output_dir.expanduser().resolve())
    out_dir = args.output_dir.expanduser().resolve()
    
    out_dir.mkdir(parents=True, exist_ok=True)

    top_rows: List[tuple[Dict[str, str], float]] = []
    if args.exp_ids:
        for exp_id in args.exp_ids:
            row = {"exp_id": str(exp_id), "frozen_mode": "B"}
            exp_dir = _find_exp_dir(sweep_dir, row)
            if exp_dir is None:
                print(f"[WARN] exp_{exp_id}: experiment folder not found")
                continue
            score = _load_experiment_score(exp_dir)
            top_rows.append((row, score))
    else:
        if args.results_csv is None:
            print("[FAIL] Provide either --results-csv or --exp-ids")
            return 1

        csv_path = args.results_csv.expanduser().resolve()
        if not csv_path.exists():
            print(f"[FAIL] CSV not found: {csv_path}")
            return 1

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
    
    print(f"\n[INFO] Preparing shortlist in {out_dir}\n")
    
    total_copied = 0
    
    for exp_idx, (row, exp_score) in enumerate(top_rows, start=1):
        exp_id = row["exp_id"]
        exp_dir = _find_exp_dir(sweep_dir, row)
        if exp_dir is None:
            print(f"[SKIP] exp_{exp_id}: experiment folder not found")
            continue

        exp_name = exp_dir.name
        gen_audio_dir = exp_dir / "generated_audio"
        
        if not gen_audio_dir.exists():
            print(f"[SKIP] {exp_name}: generated_audio not found")
            continue
        
        aggregated = _load_exp_metrics(exp_dir)
        if not aggregated:
            print(f"[SKIP] {exp_name}: no eval metrics")
            continue
        
        sorted_epochs = sorted(
            aggregated.items(),
            key=lambda x: -x[1]["score"]
        )
        
        best_epochs = [epoch for epoch, _ in sorted_epochs[:args.top_ckpt_per_exp]]
        exp_out = out_dir / f"{exp_idx:02d}_{exp_name}"
        exp_out.mkdir(parents=True, exist_ok=True)
        
        print(f"[{exp_idx}] {exp_name}")
        print(f"     Config score: {exp_score:.4f}")
        print(f"     Top checkpoints to shortlist:", end="")
        
        for epoch in best_epochs:
            print(f" epoch_{epoch:03d}", end="")
        print()
        
        count = 0
        for audio_file in sorted(gen_audio_dir.glob("*.flac")):
            epoch = _extract_epoch(audio_file.name)
            if epoch in best_epochs:
                dest = exp_out / audio_file.name
                shutil.copy2(audio_file, dest)
                count += 1
        
        print(f"     → {count} audios copied\n")
        total_copied += count
    
    print("=" * 80)
    print(f"[DONE] Shortlist ready: {total_copied} audios in {out_dir}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
