# GridGuard Data Directory

This directory stores raw, external, and processed power system telemetry data for the GridGuard research framework.

## Structure
- `external/ieee123/`: Contains the official IEEE 123-node OpenDSS test feeder definition files, acquired via `scripts/fetch_ieee123.py` with SHA-256 integrity verification.
- `processed/`: Contains processed time-series telemetry datasets (`telemetry_dataset.parquet`), graph adjacency matrices (`adjacency_matrix.npy`), and metadata artifacts (`node_metadata.json`, `attack_metadata.json`, `provenance.json`).

## Reproduction
To regenerate the processed datasets:
```bash
python scripts/generate_dataset.py --config configs/full.yaml
```
or for quick testing:
```bash
python scripts/generate_dataset.py --config configs/quick.yaml
```
