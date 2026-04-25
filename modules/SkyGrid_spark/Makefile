.PHONY: all main multiseed ablation scaling fault validation pipeline-stress placement-stress figures test

PYTHON ?= python

all: main multiseed ablation scaling fault validation pipeline-stress placement-stress figures

main:
	$(PYTHON) scripts/run_experiment.py --config configs/default.yaml --output outputs/metrics.json

multiseed:
	$(PYTHON) scripts/run_multiseed.py --config configs/default.yaml --seeds 20260928 20260929 20260930 --out outputs/multiseed.json

ablation:
	$(PYTHON) scripts/run_ablation.py --config configs/default.yaml --output outputs/ablation/ablation.json

scaling:
	$(PYTHON) scripts/run_scaling.py --config configs/scaling.yaml --output outputs/scaling/scaling.json

fault:
	$(PYTHON) scripts/run_fault.py --config configs/default.yaml --out outputs/fault/fault.json

validation:
	$(PYTHON) scripts/run_validation.py --config configs/default.yaml --out outputs/validation/validation.json

pipeline-stress:
	$(PYTHON) scripts/run_pipeline_stress.py --config configs/default.yaml --out outputs/burst_pipeline.json

placement-stress:
	$(PYTHON) scripts/run_placement_stress.py --config configs/default.yaml --out outputs/placement_stress.json

figures:
	$(PYTHON) scripts/plot_results.py --metrics outputs/metrics.json --ablation outputs/ablation/ablation.json --scaling outputs/scaling/scaling.json --outdir outputs/figs

test:
	$(PYTHON) -m pytest -q
