#!/usr/bin/env bash

set -euo pipefail

echo "================================================================================"
echo "[ROUND 2] Modal LoRA Training - Mode-focused Iteration (Linux)"
echo "================================================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv if present
if [ -f "./.venv/bin/activate" ]; then
  echo
  echo "[INFO] Activating virtual environment..."
  # shellcheck disable=SC1091
  source "./.venv/bin/activate"
fi

echo
echo "[STEP 1/3] Generate dataset with SAFE mode captions"
python scripts/prepare_modes_dataset.py \
  --dataset-dir ./Dataset \
  --output ./datasets/modos_dataset_v2.json \
  --caption-profile safe

echo
echo "[STEP 2/3] Preprocess dataset JSON to tensors (NO UI needed)"
echo "Output tensors: ./preprocessed_tensors/modal_v2"
python train.py fixed \
  --checkpoint-dir ./checkpoints \
  --model-variant turbo \
  --dataset-dir ./preprocessed_tensors/modal_v2 \
  --output-dir ./lora_output/modal_run_D \
  --preprocess \
  --dataset-json ./datasets/modos_dataset_v2.json \
  --tensor-output ./preprocessed_tensors/modal_v2

echo
echo "[STEP 3/3] Training LoRA (160 epochs, safe mode captions)"
python train.py fixed \
  --checkpoint-dir ./checkpoints \
  --model-variant turbo \
  --dataset-dir ./preprocessed_tensors/modal_v2 \
  --output-dir ./lora_output/modal_run_D \
  --adapter-type lora \
  --rank 64 \
  --alpha 128 \
  --dropout 0.05 \
  --lr 5e-5 \
  --batch-size 1 \
  --gradient-accumulation 4 \
  --epochs 160 \
  --save-every 20 \
  --optimizer-type adamw \
  --scheduler-type cosine \
  --cfg-ratio 0.15 \
  --seed 42

echo
echo "================================================================================"
echo "[SUCCESS] Round 2 training completed!"
echo "================================================================================"
echo
echo "Next steps:"
echo "1. Tensors: ./preprocessed_tensors/modal_v2"
echo "2. Checkpoints: ./lora_output/modal_run_D/"
echo "3. Priority eval: epoch 80 and 160 (Dorian/Phrygian first)"
