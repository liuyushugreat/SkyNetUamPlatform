# SkyGrid — Distributed Edge-Cloud Runtime for Neuro-Symbolic UAM Reasoning

SkyGrid is the **parallel/distributed runtime** of the SkyNetUAM
platform.  It is the companion artifact of our ICPP 2026 submission:

> *SkyGrid: Spatio-Temporal Partitioning, Cost-aware Operator
> Placement, and Asynchronous Batched Pipelines for City-Scale
> Neuro-Symbolic UAM Reasoning*

The module answers a single question that the rest of the platform
leaves open:

> *How do we schedule a hybrid neural + symbolic DAG over a real
> edge-cloud fabric so that 12 K ops/s of UAM telemetry can be
> reasoned about with 55 ms p99 latency and 8× less cross-edge
> traffic than a cloud-only baseline?*

## At a glance

| Contribution | Role | Measurable effect (default config) |
|---|---|---|
| **STP** Spatio-Temporal Partitioning | Spatial grid + FM refinement bounded by $(1{+}\gamma)\bar n$ | Edge cut 0.761 → **0.077** (10×), spatial compactness 29.4 → **12.4** |
| **COP** Cost-aware Operator Placement | Closed-form compute + transfer + queueing cost, greedy + local swap | p99 latency 84.5 ms (cloud-only) → **55.6 ms** (−34 %) |
| **ABP** Asynchronous Batched Pipeline | Micro-batching with bounded staleness + backpressure per stage | All 729 K events complete within a 60 s virtual horizon at 12 K ops/s |

End-to-end, SkyGrid delivers:

- **p50 / p95 / p99 = 32.3 / 49.3 / 55.6 ms** at 12.16 K ops/s.
- **149.7 MB** of cross-edge traffic vs. **1.22 GB** for cloud-only
  (−88 %).
- **Partition reassignment cost is amortized**: 493 boundary-cell
  moves across the whole 60 s window.
- Deterministic and CPU-only: one seed reproduces every number.

Full details and the evaluation tables are in the ICPP 2026 paper at
[`pressRequire/SkyGrid/SkyGrid_ICPP2026/skygrid_icpp2026.tex`](../../../pressRequire/SkyGrid/SkyGrid_ICPP2026).

## Repository layout

```
modules/SkyGrid/
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
cd modules/SkyGrid
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 2. Reproduce all paper results (~5–10 min on one CPU core)

```bash
# Main table (7 configurations): Table 1 of the paper
python scripts/run_experiment.py --config configs/default.yaml \
    --out outputs/metrics.json

# Ablations: STP / COP / ABP isolated
python scripts/run_ablation.py --config configs/default.yaml \
    --out outputs/ablation

# Scaling (weak / strong / entity)
python scripts/run_scaling.py --config configs/scaling.yaml \
    --out outputs/scaling

# Figures 2–5 (bars, tail-CDF, cross-edge traffic, scaling curves)
python scripts/plot_results.py --metrics outputs/metrics.json \
    --ablation outputs/ablation --scaling outputs/scaling \
    --out outputs/figs
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

Every number in the paper (tables, figures, footnotes) is a direct
output of the scripts in `scripts/` on top of the YAML configs in
`configs/`.  Seeds are pinned in each config (default `20260928`).
`SkyGridRuntime` emits `RunMetrics` dataclasses that serialize to
JSON verbatim — reviewers can diff two runs field-by-field.

## License

Apache 2.0 — same as the parent SkyNetUamPlatform repository.
