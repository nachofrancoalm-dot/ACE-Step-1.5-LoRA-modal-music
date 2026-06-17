# Run Round 2 Modal LoRA Training (Mode-focused captions)
# Windows PowerShell script
# Usage: .\run_modal_lora_training_round2.ps1

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "[ROUND 2] Modal LoRA Training - Mode-focused Iteration" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Cyan

# Activate environment if needed
$venvPath = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    Write-Host "`n[INFO] Activating virtual environment..." -ForegroundColor Yellow
    & $venvPath
}

Write-Host "`n[STEP 1/3] Generate dataset with SAFE mode captions" -ForegroundColor Green
Write-Host "Command: python scripts/prepare_modes_dataset.py --dataset-dir ./Dataset --output ./datasets/modos_dataset_v2.json --caption-profile safe" -ForegroundColor Gray
python scripts/prepare_modes_dataset.py --dataset-dir ./Dataset --output ./datasets/modos_dataset_v2.json --caption-profile safe
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Dataset generation failed" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Dataset generated with SAFE mode captions" -ForegroundColor Green

Write-Host "`n[STEP 2/3] Preprocess dataset JSON to tensors (NO UI needed)" -ForegroundColor Green
Write-Host "Output tensors: ./preprocessed_tensors/modal_v2" -ForegroundColor Gray

python train.py fixed `
    --checkpoint-dir ./checkpoints `
    --model-variant turbo `
    --dataset-dir ./preprocessed_tensors/modal_v2 `
    --output-dir ./lora_output/modal_run_D `
    --preprocess `
    --dataset-json ./datasets/modos_dataset_v2.json `
    --tensor-output ./preprocessed_tensors/modal_v2

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Preprocess failed" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Tensors generated at ./preprocessed_tensors/modal_v2" -ForegroundColor Green

Write-Host "`n[STEP 3/3] Training LoRA (160 epochs, safe mode captions)" -ForegroundColor Green
Write-Host "Configuration: r64, alpha=128, dropout=0.05, lr=5e-5, epochs=160" -ForegroundColor Gray
Write-Host "Output: ./lora_output/modal_run_D/" -ForegroundColor Gray

python train.py fixed `
    --checkpoint-dir ./checkpoints `
    --model-variant turbo `
    --dataset-dir ./preprocessed_tensors/modal_v2 `
    --output-dir ./lora_output/modal_run_D `
    --adapter-type lora `
    --rank 64 `
    --alpha 128 `
    --dropout 0.05 `
    --lr 5e-5 `
    --batch-size 1 `
    --gradient-accumulation 4 `
    --epochs 160 `
    --save-every 20 `
    --optimizer-type adamw `
    --scheduler-type cosine `
    --cfg-ratio 0.15 `
    --seed 42

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Training failed" -ForegroundColor Red
    exit 1
}

Write-Host "`n" + "=" * 80 -ForegroundColor Cyan
Write-Host "[SUCCESS] Round 2 training completed!" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Cyan

Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "1. Checkpoints saved to: ./lora_output/modal_run_D/" -ForegroundColor White
Write-Host "2. Key checkpoints to evaluate: 20, 40, 60, 80, 160" -ForegroundColor White
Write-Host "3. Run inference & manual evaluation" -ForegroundColor White
Write-Host "4. Compare with Run C epoch_160 baseline" -ForegroundColor White
