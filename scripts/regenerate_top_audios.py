#!/usr/bin/env python3
"""Regenerate audio for top N experiments only (for manual evaluation).

Usage:
    python scripts/regenerate_top_audios.py \
        --results-csv ./hparam_sweep_stage1_results/stage1_results.csv \
        --output-dir ./hparam_sweep_stage1_results \
        --api-base-url http://127.0.0.1:7860 \
        --top-n 5 \
        --prompts-block E \
        --seeds 1111 2222
"""

import argparse
import csv
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple


def _run(cmd: List[str], cwd: Path) -> Tuple[int, str]:
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


def _build_exp_dir_from_row(output_dir: Path, row: dict) -> Path:
    exp_id = row["exp_id"]
    frozen_mode = row["frozen_mode"]
    rank = int(float(row["rank"]))
    alpha = int(float(row["alpha"]))
    lr = row["lr"]
    dropout = float(row["dropout"])
    
    exp_name = f"exp_{int(exp_id):03d}_{frozen_mode}_r{rank}_a{alpha}_lr{lr}_do{dropout}"
    return output_dir / exp_name


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate audio for top experiments")
    parser.add_argument("--results-csv", type=Path, required=True, help="Path to stage1_results.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="Sweep output directory")
    parser.add_argument("--api-base-url", type=str, default="http://127.0.0.1:7860",
                       help="ACE-Step API base URL")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top experiments")
    parser.add_argument("--prompts-block", type=str, default="E", help="Prompts block")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1111, 2222], help="Seeds")
    
    args = parser.parse_args()
    
    csv_path = args.results_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    
    if not csv_path.exists():
        print(f"[FAIL] CSV not found: {csv_path}")
        return 1
    
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("training_status", "").upper() == "SUCCESS":
                try:
                    score = float(row.get("auto_score_0_1", "-1"))
                    if score >= 0:
                        rows.append((row, score))
                except Exception:
                    pass
    
    rows.sort(key=lambda x: -x[1])
    top_rows = rows[:args.top_n]
    
    if not top_rows:
        print("[FAIL] No valid experiments found")
        return 1
    
    print(f"[INFO] Will regenerate audio for top {len(top_rows)} experiments\n")
    
    for row, score in top_rows:
        exp_id = row["exp_id"]
        exp_dir = _build_exp_dir_from_row(output_dir, row)
        
        if not exp_dir.exists():
            print(f"[WARN] exp {exp_id}: folder not found: {exp_dir}")
            continue
        
        generated_audio = exp_dir / "generated_audio"
        manifest_csv = exp_dir / "modal_batch_manifest.csv"
        
        print(f"[INFO] exp {exp_id} (score={score:.4f}): running inference...")
        
        cmd = [
            "python",
            "scripts/modal_batch_infer.py",
            "--api-base-url",
            args.api_base_url,
            "--checkpoints-root",
            str(exp_dir),
            "--prompts-block",
            args.prompts_block,
            "--seeds",
            *[str(s) for s in args.seeds],
            "--output-csv",
            str(manifest_csv),
            "--copy-audio-dir",
            str(generated_audio),
        ]
        
        code, output = _run(cmd, cwd=Path.cwd())
        if code != 0:
            print(f"[FAIL] exp {exp_id}: inference failed")
            print(output[-500:])
            continue
        
        audio_count = len(list(generated_audio.glob("*")))
        print(f"[OK] exp {exp_id}: {audio_count} audios generated at {generated_audio}\n")
    
    print("[SUMMARY]")
    print("Audios ready for evaluation in:")
    for row, score in top_rows:
        exp_id = row["exp_id"]
        exp_dir = _build_exp_dir_from_row(output_dir, row)
        generated_audio = exp_dir / "generated_audio"
        print(f"  {generated_audio}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
