# Full Benchmark Reproduction Script (PowerShell)
$ErrorActionPreference = "Stop"

Write-Host "=== Step 0: Fetching IEEE 123-Node Test Feeder Files ===" -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" scripts/fetch_ieee123.py

Write-Host "=== Step 1: Generating Full IEEE-123 Benchmark Dataset ===" -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" scripts/generate_dataset.py --config configs/full.yaml

Write-Host "=== Step 2: Training Baselines (B0 to B6) ===" -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" scripts/train_baselines.py --config configs/full.yaml

Write-Host "=== Step 3: Training Proposed Physics-Guided GTAE (B7) ===" -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" scripts/train_proposed.py --config configs/full.yaml

Write-Host "=== Step 4: Running Component Ablations (A1 to A6) ===" -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" scripts/run_ablations.py --config configs/full.yaml

Write-Host "=== Step 5: Running Robustness Matrix Evaluation ===" -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" scripts/run_robustness.py --config configs/full.yaml

Write-Host "=== Step 6: Performing Benchmark Evaluation & Hypothesis Testing ===" -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" scripts/evaluate.py --config configs/full.yaml

Write-Host "=== Step 7: Generating Publication Figures (1-11) ===" -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" scripts/make_figures.py --config configs/full.yaml

Write-Host "=== Step 8: Generating Publication Tables (1-8) ===" -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" scripts/make_tables.py --config configs/full.yaml

Write-Host "=== Full Benchmark Reproduction Completed Successfully ===" -ForegroundColor Green
