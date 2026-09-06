#!/usr/bin/env bash
set -e

echo "=== Step 1: Generating Quick Dataset ==="
python scripts/generate_dataset.py --config configs/quick.yaml

echo "=== Step 2: Training Baselines ==="
python scripts/train_baselines.py --config configs/quick.yaml

echo "=== Step 3: Training Proposed Model ==="
python scripts/train_proposed.py --config configs/quick.yaml

echo "=== Step 4: Running Component Ablations ==="
python scripts/run_ablations.py --config configs/quick.yaml

echo "=== Step 5: Running Robustness Evaluation ==="
python scripts/run_robustness.py --config configs/quick.yaml

echo "=== Step 6: Evaluating Metrics ==="
python scripts/evaluate.py --config configs/quick.yaml

echo "=== Step 7: Generating Figures ==="
python scripts/make_figures.py --config configs/quick.yaml

echo "=== Step 8: Generating Tables ==="
python scripts/make_tables.py --config configs/quick.yaml

echo "=== Quick Verification Completed Successfully ==="
