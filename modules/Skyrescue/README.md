# SkyRescue

SkyRescue is the paper module for large-scale low-altitude emergency traffic command.  It contains the deterministic scheduler, symbolic grounding and audit ablations, weak-signal fault challenge scorer, security challenge scorer, and scripts for generating the synthetic SkyRescue-Bench datasets.

## What is included

- `skyrescue/benchmark.py`: SkyRescue runtime and baselines (`greedy`, `cp_sat`, `no_symbol_grounding`, `no_audit`, `full_replan`, `skyrescue`).
- `skyrescue/fault_detection.py`: online weak-signal detector that does not read fault labels during inference.
- `skyrescue/security.py`: deterministic authorization boundary for security challenge evaluation.
- `scripts/generate_dataset.py`: synthetic emergency-traffic benchmark generator.
- `scripts/generate_security_challenges.py`: authorization challenge generator.
- `scripts/generate_fault_challenge.py`: weak-signal partial-observability fault challenge generator.
- `scripts/run_skyrescue_benchmark.py`: multi-dataset evaluator and 10-seed summary writer.
- `scripts/run_security_challenges.py`: security challenge scorer.
- `scripts/run_fault_challenge.py`: fault challenge scorer.

Large generated datasets are intentionally excluded from this Git module. Keep them in the paper workspace or publish them separately through an archival dataset host.

## Quick start

```bash
cd modules/Skyrescue
python -m venv .venv-skyrescue
source .venv-skyrescue/bin/activate
pip install -r requirements.txt
```

Generate a small synthetic benchmark:

```bash
python scripts/generate_dataset.py --config configs/small.json --output /tmp/skyrescue-small
python scripts/validate_dataset.py --dataset /tmp/skyrescue-small
```

Run the main evaluator:

```bash
python scripts/run_skyrescue_benchmark.py \
  --datasets /tmp/skyrescue-small \
  --output-dir /tmp/skyrescue-results/small_run
```

Run the two challenge scorers:

```bash
python scripts/generate_security_challenges.py --output /tmp/skyrescue-security-challenge
python scripts/run_security_challenges.py --dataset /tmp/skyrescue-security-challenge --output /tmp/skyrescue-results/security_challenge_v1.json

python scripts/generate_fault_challenge.py --output /tmp/skyrescue-fault-challenge
python scripts/run_fault_challenge.py --dataset /tmp/skyrescue-fault-challenge --output /tmp/skyrescue-results/fault_challenge_v1.json
```

## Current paper-scale results

The paper workspace currently contains a 10-seed synthetic evaluation under `SkyRescue-Bench/results/skyrescue_experiments_10seed`.  Summary values should be reported as synthetic benchmark evidence, not real flight evidence.

Key values from the current 10-seed run:

| Method | Completion | On-time rate | Conflict rate | Evidence completeness |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 0.7100 ± 0.1252 | 0.0500 ± 0.0247 | 0.3562 ± 0.2127 | 0.2015 ± 0.0508 |
| CP-SAT assignment + scheduler | 0.9458 ± 0.0087 | 0.5975 ± 0.0890 | 0.0000 ± 0.0000 | 0.9129 ± 0.0135 |
| No symbolic grounding | 0.2913 ± 0.1391 | 0.0050 ± 0.0021 | 0.0000 ± 0.0000 | 0.2897 ± 0.1390 |
| No audit | 1.0000 ± 0.0000 | 0.9340 ± 0.0155 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Full replan | 1.0000 ± 0.0000 | 0.9090 ± 0.0192 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| SkyRescue | 1.0000 ± 0.0000 | 0.9340 ± 0.0155 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 |

## Reproducibility notes

- All benchmark data are synthetic and generated from fixed seeds.
- Fault labels are withheld from online scheduling and detection, then opened only for offline scoring.
- The CP-SAT baseline is a centralized resource-assignment baseline followed by the shared reservation scheduler; it is not a proof of global optimality for the full traffic-control problem.
- The security challenge evaluates deterministic policy coverage rather than operational flight safety.
