#!/usr/bin/env bash
set -e

echo "=== Step 0: Fetching IEEE 123-Node Test Feeder Files ==="
python scripts/fetch_ieee123.py

echo "=== Step 1: Generating Full IEEE-123 Benchmark Dataset ==="
python scripts/generate_dataset.py --config configs/full.yaml

echo "=== Step 2: Training Baselines (B0 to B6) ==="
python scripts/train_baselines.py --config configs/full.yaml

echo "=== Step 3: Training Proposed Physics-Guided GTAE (B7) ==="
python scripts/train_proposed.py --config configs/full.yaml

echo "=== Step 4: Running Component Ablations (A1 to A6) ==="
python scripts/run_ablations.py --config configs/full.yaml

echo "=== Step 5: Running Robustness Matrix Evaluation ==="
python scripts/run_robustness.py --config configs/full.yaml

echo "=== Step 6: Performing Benchmark Evaluation & Hypothesis Testing ==="
python scripts/evaluate.py --config configs/full.yaml

echo "=== Step 7: Generating Publication Figures (1-11) ==="
python scripts/make_figures.py --config configs/full.yaml

echo "=== Step 8: Generating Publication Tables (1-8) ==="
python scripts/make_tables.py --config configs/full.yaml

echo "=== Full Benchmark Reproduction Completed Successfully ==="
