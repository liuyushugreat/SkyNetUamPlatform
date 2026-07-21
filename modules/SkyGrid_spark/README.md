# SkyGrid — Distributed Edge-Cloud Runtime for Neuro-Symbolic UAM Reasoning

SkyGrid is the **parallel/distributed runtime** evaluated in the
IEEE HPCC 2026 submission:

> *SkyGrid: A Distributed Edge-Cloud Runtime for Hardware-Aware
> Partitioning and Placement of Hybrid Neural-Symbolic Pipelines*

The module answers a single question that the rest of the platform
leaves open:

> *How do we schedule a hybrid neural + symbolic DAG over an edge-cloud
> fabric so that city-scale UAM telemetry stays near a 100 ms p99 SLO
> while reducing cross-edge traffic and remaining robust to degraded
> edge capacity?*

## At a glance

| Contribution | Role | Measurable effect (default config) |
|---|---|---|
| **STP** Spatio-Temporal Partitioning | Spatial grid + FM refinement bounded by $(1{+}\gamma)\bar n$ | Edge cut 0.680 -> **0.077** vs. LDG and cross-edge traffic **397.8 MB -> 173.9 MB** |
| **COP-H** Hardware-aware Operator Placement | Closed-form compute + transfer + state-tier + queueing cost, greedy + local swap | p99 latency **~100.5 ms -> 72.1 ms** vs. static placement |
| **ABP** Asynchronous Batched Pipeline | Micro-batching with bounded staleness + hysteretic backpressure per stage | Provides bounded overlap/backpressure; sync is also reported because it is faster at unsaturated M-load |

End-to-end, SkyGrid delivers:

- **p50 / p95 / p99 = 48.42 / 65.31 / 72.14 ms** at 13.94 K ops/s
  in the three-seed M-regime summary.
- **173.9 MB** of cross-edge traffic vs. **397.8 MB** for LDG+COP-H
  and **~1.68 GB** for static-placement baselines.
- **Partition reassignment cost is amortized**: 493 boundary-cell
  moves across the whole 60 s window.
- Deterministic and CPU-only: one seed reproduces every number.

Full details and the evaluation tables are in the IEEE HPCC 2026 paper at
[`pressRequire/SkyGrid_spark/skygrid_spark.tex`](../../../pressRequire/SkyGrid_spark/skygrid_spark.tex).

## Repository layout

```
modules/SkyGrid_spark/
├── configs/                  # YAML configs (default / ablation / scaling)
├── scripts/
│   ├── run_experiment.py     # main Table-1 runs (7 configs, batched)
│   ├── run_ablation.py       # component-level ablations
│   ├── run_scaling.py        # weak / strong / entity scaling
│   ├── plot_results.py       # regenerates every figure from metrics.json
│   └── make_illustrations.py # DAG + architecture vector diagrams
├── skygrid/                  # Python package
│   ├── workload/             # city-scale UAM event generator (deterministic)
│   ├── partition/            # STP + LDG/hash/random baselines
│   ├── placement/            # COP + cost_model + static/ILP/random baselines
│   ├── pipeline/             # ABP + sync baseline (micro-batcher + DAG)
│   ├── simulator/            # Discrete-event simulator (nodes + links)
│   ├── runtime/              # SkyGridRuntime orchestrator
│   └── telemetry/            # Tracer + metrics (JSONL-friendly)
├── tests/                    # 18 pytest cases; all green on pure CPU
└── outputs/                  # metrics.json + ablation/ + scaling/ + figs/
```

## One-click reproduction

The artifact is pure Python + NumPy.  No GPU, no network, no API keys.

### 1. Install

```bash
cd modules/SkyGrid_spark
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 2. Reproduce all paper results (~5–10 min on one CPU core)

```bash
# Main table (8 configurations): Table 2 of the paper
python scripts/run_experiment.py --config configs/default.yaml \
    --output outputs/metrics.json

# Three-seed aggregate for Table 2
python scripts/run_multiseed.py --config configs/default.yaml \
    --seeds 20260928 20260929 20260930 \
    --out outputs/multiseed.json

# Ablations: STP / COP-H / ABP / state locality isolated
python scripts/run_ablation.py --config configs/default.yaml \
    --output outputs/ablation/ablation.json

# Scaling (weak / strong / entity)
python scripts/run_scaling.py --config configs/scaling.yaml \
    --output outputs/scaling/scaling.json

# Fault/degradation and cost-model validation
python scripts/run_fault.py --config configs/default.yaml \
    --out outputs/fault/fault.json
python scripts/run_validation.py --config configs/default.yaml \
    --out outputs/validation/validation.json

# Pipeline stress: ABP vs synchronous execution under bursty load
python scripts/run_pipeline_stress.py --config configs/default.yaml \
    --out outputs/burst_pipeline.json

# Placement stress: state-tier-aware COP-H vs tier-blind LocAware
python scripts/run_placement_stress.py --config configs/default.yaml \
    --out outputs/placement_stress.json

# Figures 2–5 (bars, tail-CDF, cross-edge traffic, scaling curves)
python scripts/plot_results.py --metrics outputs/metrics.json \
    --ablation outputs/ablation/ablation.json \
    --scaling outputs/scaling/scaling.json \
    --outdir outputs/figs
```

### 3. Run the unit-test suite (18 cases)

```bash
pytest -q
```

Expected output: `18 passed` in ~12 s.

## Extending SkyGrid

* **New partitioner.**  Subclass `skygrid.partition.base.Partitioner`,
  register it in `skygrid.partition.__init__.PARTITIONERS`.  Add a
  row to `scripts/run_experiment.py` and re-run to compare against
  STP / LDG / hash / random.
* **New placer.**  Same idea under
  `skygrid.placement.{base,solver}`; the cost model is
  parameterized over four fabric numbers (TFLOPS, latency,
  bandwidth, jitter), so heterogeneous edges need no code changes.
* **New pipeline.**  Implement `PipelineRunner.step(event)` under
  `skygrid/pipeline/` — ABP and Sync are 150-line reference
  implementations you can use as templates.

## Reproducibility guarantee

Every number in the paper is generated by scripts under `scripts/`
from YAML configs under `configs/`.  Seeds are pinned in each config
(default `20260928`), and `SkyGridRuntime` emits `RunMetrics`
dataclasses that serialize to JSON verbatim, so reviewers can diff two
runs field-by-field.

## License

This project is licensed under the Apache License 2.0.
