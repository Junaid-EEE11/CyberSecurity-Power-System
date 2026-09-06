# Quick Smoke-Test Reproduction Script (PowerShell)
$ErrorActionPreference = "Stop"

Write-Host "=== Step 1: Generating Quick Dataset ===" -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" scripts/generate_dataset.py --config configs/quick.yaml

Write-Host "=== Step 2: Training Baselines ===" -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" scripts/train_baselines.py --config configs/quick.yaml

Write-Host "=== Step 3: Training Proposed Model ===" -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" scripts/train_proposed.py --config configs/quick.yaml

Write-Host "=== Step 4: Running Component Ablations ===" -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" scripts/run_ablations.py --config configs/quick.yaml

Write-Host "=== Step 5: Running Robustness Evaluation ===" -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" scripts/run_robustness.py --config configs/quick.yaml

Write-Host "=== Step 6: Evaluating Metrics ===" -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" scripts/evaluate.py --config configs/quick.yaml

Write-Host "=== Step 7: Generating Figures ===" -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" scripts/make_figures.py --config configs/quick.yaml

Write-Host "=== Step 8: Generating Tables ===" -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" scripts/make_tables.py --config configs/quick.yaml

Write-Host "=== Quick Verification Completed Successfully ===" -ForegroundColor Green
