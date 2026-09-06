.PHONY: help setup fetch test quick data baselines proposed ablations robustness evaluate figures tables reproduce full

PYTHON := python
CONFIG_QUICK := configs/quick.yaml
CONFIG_FULL := configs/full.yaml

help:
	@echo "GridGuard Research Repository Commands:"
	@echo "  make setup       : Install dependencies and package in editable mode"
	@echo "  make fetch       : Download and verify IEEE 123-bus test feeder files"
	@echo "  make test        : Run full test suite with pytest"
	@echo "  make quick       : Run fast end-to-end smoke test"
	@echo "  make data        : Generate simulation dataset"
	@echo "  make baselines   : Train all classical and neural baselines"
	@echo "  make proposed    : Train proposed physics-guided graph-temporal model"
	@echo "  make ablations   : Run ablation studies"
	@echo "  make robustness  : Run robustness evaluation (noise/missingness)"
	@echo "  make evaluate    : Evaluate models and compute metrics"
	@echo "  make figures     : Generate all publication figures"
	@echo "  make tables      : Generate all publication tables"
	@echo "  make full        : Run full reproduction pipeline"

setup:
	$(PYTHON) -m pip install -e .

fetch:
	$(PYTHON) scripts/fetch_ieee123.py

test:
	$(PYTHON) -m pytest -v

quick:
	$(PYTHON) scripts/generate_dataset.py --config $(CONFIG_QUICK)
	$(PYTHON) scripts/train_baselines.py --config $(CONFIG_QUICK)
	$(PYTHON) scripts/train_proposed.py --config $(CONFIG_QUICK)
	$(PYTHON) scripts/evaluate.py --config $(CONFIG_QUICK)
	$(PYTHON) scripts/make_figures.py --config $(CONFIG_QUICK)
	$(PYTHON) scripts/make_tables.py --config $(CONFIG_QUICK)

data:
	$(PYTHON) scripts/generate_dataset.py --config $(CONFIG_FULL)

baselines:
	$(PYTHON) scripts/train_baselines.py --config $(CONFIG_FULL)

proposed:
	$(PYTHON) scripts/train_proposed.py --config $(CONFIG_FULL)

ablations:
	$(PYTHON) scripts/run_ablations.py --config $(CONFIG_FULL)

robustness:
	$(PYTHON) scripts/run_robustness.py --config $(CONFIG_FULL)

evaluate:
	$(PYTHON) scripts/evaluate.py --config $(CONFIG_FULL)

figures:
	$(PYTHON) scripts/make_figures.py --config $(CONFIG_FULL)

tables:
	$(PYTHON) scripts/make_tables.py --config $(CONFIG_FULL)

full: data baselines proposed ablations robustness evaluate figures tables
