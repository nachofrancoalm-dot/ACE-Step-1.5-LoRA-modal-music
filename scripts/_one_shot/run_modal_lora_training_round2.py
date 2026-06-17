#!/usr/bin/env python3
"""
Run 2nd training iteration for Modal LoRA.
Mode-focused captions + reduced epochs for faster iteration.

Usage (from project root):
    uv run python scripts/run_modal_lora_training_round2.py

This script:
1. Generates dataset with mode-focused captions (prepare_modes_dataset.py)
2. Preprocesses dataset JSON to tensors (CLI, no UI required)
3. Executes train.py with optimized hyperparameters for 2nd iteration
3. Saves to lora_output/modal_run_D (run C was first iteration)

Configuration:
- Same hyperparams as Run C but epochs=100 (not 200) for faster iteration
- Safe caption profile by default: avoids fixing style/tonic/BPM
- Save checkpoint every 20 epochs for evaluation flexibility
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd: list[str], description: str) -> int:
    """Execute command and return exit code."""
    print(f"\n{'='*80}")
    print(f"[STEP] {description}")
    print(f"{'='*80}")
    print(f"$ {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    if result.returncode != 0:
        print(f"\n[ERROR] {description} failed with code {result.returncode}")
        return result.returncode
    print(f"\n[OK] {description} completed.")
    return 0

def main():
    # Step 1: Generate dataset with mode-focused captions
    print("\n" + "="*80)
    print("[ROUND 2] Modal LoRA Training - Mode-focused Iteration")
    print("="*80)
    
    exit_code = run_command(
        [
            sys.executable,
            "scripts/prepare_modes_dataset.py",
            "--dataset-dir", "./Dataset",
            "--output", "./datasets/modos_dataset_v2.json",
            "--caption-profile", "safe",
        ],
        "Generate dataset with SAFE mode captions"
    )
    
    if exit_code != 0:
        return exit_code

    # Step 2: Preprocess JSON dataset into tensor cache
    exit_code = run_command(
        [
            sys.executable,
            "train.py", "fixed",
            "--checkpoint-dir", "./checkpoints",
            "--model-variant", "turbo",
            "--dataset-dir", "./preprocessed_tensors/modal_v2",
            "--output-dir", "./lora_output/modal_run_D",
            "--preprocess",
            "--dataset-json", "./datasets/modos_dataset_v2.json",
            "--tensor-output", "./preprocessed_tensors/modal_v2",
        ],
        "Preprocess dataset JSON to tensors (modal_v2)"
    )

    if exit_code != 0:
        return exit_code
    
    # Step 3: Train LoRA with safe mode captions (same config, reduced epochs)
    exit_code = run_command(
        [
            sys.executable,
            "train.py", "fixed",
            "--checkpoint-dir", "./checkpoints",
            "--model-variant", "turbo",
            "--dataset-dir", "./preprocessed_tensors/modal_v2",
            "--output-dir", "./lora_output/modal_run_D",
            "--adapter-type", "lora",
            "--rank", "64",
            "--alpha", "128",
            "--dropout", "0.05",
            "--lr", "5e-5",
            "--batch-size", "1",
            "--gradient-accumulation", "4",
            "--epochs", "160",  # CHANGED: 200 -> 160 for faster iteration
            "--save-every", "20",
            "--optimizer-type", "adamw",
            "--scheduler-type", "cosine",
            "--cfg-ratio", "0.15",
            "--seed", "42",
        ],
        "Train LoRA with SAFE mode captions (160 epochs)"
    )
    
    if exit_code == 0:
        print("\n" + "="*80)
        print("[SUCCESS] Round 2 training completed!")
        print("="*80)
        print("\nNext steps:")
        print("0. Tensors generated at: ./preprocessed_tensors/modal_v2")
        print("1. Checkpoints saved to: ./lora_output/modal_run_D/")
        print("2. Key checkpoints to evaluate: 20, 40, 60, 80, 160")
        print("3. Run inference & manual evaluation (compare with Run C epoch_160)")
        print("\nExpected timeline:")
        print("- Training: 4-5 hours")
        print("- Evaluation: 2-3 hours")
        print("- Decision & documentation: 1 hour")
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main())
