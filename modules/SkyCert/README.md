# SkyCert: Conformal and Martingale-Based Runtime Security Assurance for Neuro-Symbolic Risk Reasoning in Urban Air Mobility

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Conference: IEEE CSCloud 2026](https://img.shields.io/badge/Conference-IEEE_CSCloud_2026-green.svg)](https://www.cloud-conf.net/cscloud/2026/cscloud/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Runtime: CPU-only](https://img.shields.io/badge/runtime-CPU--only-lightgrey.svg)]()
[![Reproducible: seed 20260417](https://img.shields.io/badge/seed-20260417-informational)]()

> **IEEE CSCloud 2026 Reviewers — one-click reproduction:**
> ```bash
> cd modules/SkyCert
> bash run.sh          # Linux/macOS (≈2 min, CPU-only)
> # .\run.ps1          # Windows PowerShell
> ```
> The script installs dependencies, runs 9 unit tests, reproduces the 5-threat main experiment, the 4-variant ablation, the baseline comparison, the extension experiments (λ-sweep, attack-strength sweep, failure cases, MLP backbone), the 5-seed aggregation (Tables 1/2/3 mean±std), and renders the five paper figures (`outputs/figs/`). The script also prints a summary table with the numbers cited in the paper. No GPU, API key, or network access is required.

---

## Overview

**SkyCert** is the *assurance layer* of `SkyNetUamPlatform`. It wraps an existing neuro-symbolic risk reasoner (neural scorer + symbolic rule engine over a UAM knowledge-graph slice) with three complementary mechanisms that turn opaque risk scores into **auditable, certification-friendly decisions**:

1. **Conformal Risk Sets** — split-conformal prediction with an Adaptive Prediction Sets (APS) nonconformity score; converts raw softmax outputs into prediction sets with a finite-sample marginal-coverage guarantee (`1 − α`).
2. **Martingale Monitoring** — an online test-martingale over a *hybrid* nonconformity stream (confidence slack + standardized input drift) driven by a simple-jumper betting function; provides anytime-valid sequential evidence against the exchangeability hypothesis.
3. **Abstention / Alert / Escalation Policy** — a decision-gating layer that refuses to emit a confident answer when (i) the risk set is uninformative, (ii) the martingale breaches a pre-registered threshold, or (iii) the rule trace disagrees with the neural prediction.

Every decision is persisted as a **machine-readable audit artifact** (JSON Lines) containing the input operation ID, the symbolic rule trace, the conformal prediction set, the martingale trajectory, and the final policy verdict. These artifacts are designed to back an offline certification argument for UAM operational authorizations.

This module is the companion artifact for the IEEE CSCloud 2026 submission (track: *Security of Artificial Intelligence enabled computing*):

> *SkyCert: Conformal and Martingale-Based Runtime Security Assurance for Neuro-Symbolic Risk Reasoning in Urban Air Mobility.*

Following the SkyNetUamPlatform contribution policy, **no paper PDF/TeX source, internal safety-case document, or proprietary KG slice is committed to this repository** — only the open-source module and the default reviewer configuration (`configs/default.yaml`).

---

## Threat Model (short)

SkyCert explicitly defends against four attacker/environment classes that frequently appear in UAM safety cases:

| ID | Threat | What happens | Where simulated |
|----|--------|--------------|-----------------|
| **T1** | Knowledge-Graph Corruption | Symbolic rule deltas are flipped between low- and high-risk classes | `data/threats.py::corrupt_rules` |
| **T2** | Rule Poisoning | Adversarial rules injected into the rule base | `data/threats.py::inject_rule_noise` |
| **T3** | Feature Manipulation | Bounded `ℓ∞` perturbation of operation features | `data/threats.py::perturb_features` |
| **T4** | Covariate Shift | Fleet-wide degraded sensing + weather/traffic shift | `data/threats.py::shift_covariates` |

The neural scorer and the rule engine are both treated as *untrusted* — SkyCert guarantees coverage **regardless** of how they were trained, and signals whenever the calibration assumption is violated.

---

## Repository Structure

```
modules/SkyCert/
├── README.md                 ← You are here (reviewer entry point)
├── run.sh / run.ps1          ← One-click reproduction (Linux/macOS / Windows)
├── pyproject.toml            ← Package metadata (skycert, Apache-2.0)
├── requirements.txt          ← Pinned runtime + dev dependencies
├── .gitignore                ← Excludes outputs/, caches, internal configs
│
├── configs/
│   └── default.yaml          ← Reviewer-facing default config (all numbers in paper)
│
├── skycert/                  ← Library code (import `skycert`)
│   ├── __init__.py
│   ├── config.py             ← Typed dataclasses + YAML loader
│   ├── utils.py              ← softmax, one-hot, JSONL writer, safe-json
│   ├── metrics.py            ← Coverage, ECE, critical-error, detection metrics
│   ├── pipeline.py           ← SkyCertPipeline: end-to-end orchestrator
│   │
│   ├── data/                 ← Synthetic UAM dataset + threat injection
│   │   ├── synthetic.py      ←   make_uam_dataset (features, ordinal risk label)
│   │   └── threats.py        ←   T1 KG corruption, T2 rule poison,
│   │                             T3 feature attack, T4 covariate shift
│   │
│   ├── base/                 ← Neuro-symbolic base reasoner
│   │   ├── neural.py         ←   NeuralRiskScorer (class-balanced multinomial LR)
│   │   ├── symbolic.py       ←   SymbolicRuleEngine (feature predicates + logit delta)
│   │   └── neuro_symbolic.py ←   NeuroSymbolicRiskReasoner (fused scorer)
│   │
│   └── assurance/            ← SkyCert core
│       ├── conformal.py      ←   ConformalRiskSet (APS / LAC)
│       ├── martingale.py     ←   MartingaleMonitor + SimpleJumperBetting
│       ├── policy.py         ←   AssurancePolicy (ACCEPT / ABSTAIN / ALERT / ESCALATE)
│       └── audit.py          ←   AuditLogger (JSONL artifacts)
│
├── scripts/                  ← Experiment entry points (all CPU-only, ≈2 min total)
│   ├── run_experiment.py     ←   Main: 5 threat scenarios → metrics.json + audit/
│   ├── run_ablation.py       ←   Ablation: 4 variants under T4 → ablation.json
│   ├── run_baselines.py      ←   Baseline comparison (MSP, entropy, conformal-only) → baselines.json
│   ├── run_extensions.py     ←   λ-sweep + β3/β4 sweeps + failure cases + MLP → extensions.json
│   ├── run_multi_seed.py     ←   5-seed aggregation of (1)+(2)+(3) → multi_seed.json
│   ├── print_summary.py      ←   Pretty-print multi-seed + extensions tables for reviewers
│   └── plot_results.py       ←   Renders paper figures to outputs/figs/ (incl. Pareto,
│                                 λ-sweep, attack-strength-sweep)
│
└── tests/                    ← pytest suite (9 tests, runs in <3 s)
    ├── test_conformal.py     ←   Marginal coverage, top-1 inclusion
    ├── test_martingale.py    ←   E[M_t] ≈ 1 under H0, alerts under injected shift
    └── test_policy.py        ←   Decision matrix: ACCEPT/ABSTAIN/ALERT/ESCALATE
```

---

## Prerequisites

- **Python 3.10+** (tested on 3.10, 3.11, 3.12)
- No GPU required; all experiments run in ~30 s on a single modern CPU core
- No API keys, no network access, no external data download
- Dependencies: `numpy`, `scipy`, `pyyaml`, `tqdm`, `matplotlib`, `pytest` (all pinned in `requirements.txt`)

---

## Quick Start

```bash
git clone https://github.com/liuyushugreat/SkyNetUamPlatform.git
cd SkyNetUamPlatform/modules/SkyCert

pip install -r requirements.txt

# One-click: tests + main experiment + ablation + figures + summary
bash run.sh          # Linux/macOS
# .\run.ps1          # Windows PowerShell
```

Expected artifacts (all under `outputs/`, gitignored by design):

| File | What it contains |
|------|-----------------|
| `outputs/metrics.json` | 5 threat scenarios × {coverage, CRITICAL-class coverage, avg set size, top-1 accuracy, critical-class miss rate before/after abstention, FP/FN on CRITICAL, abstain/alert/escalation rates, martingale max, detection delay, false-alarm rate, avg decision latency, throughput} |
| `outputs/ablation.json` | 4 ablation variants under the covariate-shift threat |
| `outputs/baselines.json` | MSP, entropy, conformal-only, full SkyCert under matched abstention |
| `outputs/extensions.json` | λ-sweep, β3/β4 attack-strength sweeps, 3 failure cases, MLP backbone |
| `outputs/multi_seed.json` | Mean/std/min/max/count over 5 seeds for experiment+ablation+baselines |
| `outputs/audit/audit_<scenario>.jsonl` | Per-decision audit artifacts (1 JSON object per UAM operation) |
| `outputs/audit_ablation/<variant>.jsonl` | Per-decision audit artifacts for each ablation variant |
| `outputs/figs/coverage_vs_threat.pdf` | Empirical coverage under every threat |
| `outputs/figs/critical_error.pdf`     | Critical-class miss rate before vs. after abstention |
| `outputs/figs/martingale_max.pdf`     | Max martingale value per threat (log scale) |
| `outputs/figs/pareto.pdf`             | Abstention–safety Pareto curve |
| `outputs/figs/lambda_sweep.pdf`       | Appendix: critical miss / abstain vs. λ_drift |
| `outputs/figs/attack_strength_sweep.pdf` | Appendix: critical miss / abstain vs. β3 and β4 |

---

## Step-by-Step Reproduction (if `run.sh` is not an option)

```bash
# 1. dependencies
pip install -r requirements.txt

# 2. unit tests (9 tests, <3 s)
python -m pytest tests -q

# 3. main 5-threat experiment  →  outputs/metrics.json, outputs/audit/
python -m scripts.run_experiment --config configs/default.yaml

# 4. ablation study             →  outputs/ablation.json, outputs/audit_ablation/
python -m scripts.run_ablation   --config configs/default.yaml

# 5. baseline comparison        →  outputs/baselines.json
python -m scripts.run_baselines  --config configs/default.yaml

# 6. extension experiments      →  outputs/extensions.json
#    (λ-sweep, β3/β4 sweeps, 3 failure-case audit records, MLP backbone)
python -m scripts.run_extensions --config configs/default.yaml

# 7. 5-seed aggregation         →  outputs/multi_seed.json
#    (mean±std for Tables 1/2/3; reruns 3–5 across seeds {20260417, 1, 2, 3, 4})
python -m scripts.run_multi_seed --config configs/default.yaml

# 8. paper figures              →  outputs/figs/*.pdf
#    (Coverage vs. threat, martingale, critical error, Pareto curve,
#     λ-sweep, attack-strength sweep)
python -m scripts.plot_results   --config configs/default.yaml

# 9. pretty-print summary (reviewer convenience)
python scripts/print_summary.py
```

All stochastic components are seeded from `configs/default.yaml` (`seed: 20260417`), including a deterministic per-threat seed derived from `sha256(threat_name)`. Re-running on the same Python/NumPy version reproduces the same safety and calibration metrics bit-for-bit; the timing fields in `metrics.json` vary modestly with host load. Steps 3–8 complete in ≈2 minutes on a modern single CPU core.

---

## Paper-to-Code Mapping

| Paper section | Code / script | What it reproduces |
|---------------|---------------|--------------------|
| §II-C Threat Model (T1–T4) | `skycert/data/threats.py` | Injection procedures for KG corruption, rule poisoning, feature attack, covariate shift |
| §II-A Neuro-Symbolic Base | `skycert/base/{neural,symbolic,neuro_symbolic}.py` | Class-balanced neural scorer + rule engine with audit trace |
| §IV-B Conformal Risk Sets | `skycert/assurance/conformal.py` | APS & LAC nonconformity, split calibration, coverage guarantee |
| §IV-C Martingale Monitor | `skycert/assurance/martingale.py` | Simple-jumper test-martingale with warm-start from calibration |
| §IV-C Hybrid Nonconformity | `skycert/pipeline.py::_nonconformity` | `(1 − max_prob) + L2-drift` in standardized feature space |
| §IV-D Decision Policy | `skycert/assurance/policy.py` | ACCEPT / ABSTAIN / ALERT / ESCALATE matrix |
| §IV-E Audit Artifacts | `skycert/assurance/audit.py` | JSONL per-decision records (inputs, rule trace, set, martingale) |
| §V Implementation | `pyproject.toml`, `requirements.txt` | Pinned deterministic environment |
| §VII Main Experiment (Table I, Fig. 2–4) | `scripts/run_experiment.py` | 5 threat scenarios × full metric panel |
| §VIII Ablation (Table II) | `scripts/run_ablation.py` | 4 variants: `no_conformal`, `no_martingale`, `no_abstention`, `full` |
| §VIII-A Baselines (Table III) | `scripts/run_baselines.py` | MSP, entropy, conformal-only; abstention matched per seed |
| §VII–§VIII Multi-seed aggregation (mean±std) | `scripts/run_multi_seed.py` | Reruns experiment/ablation/baselines across seeds `{20260417,1,2,3,4}` |
| Appendix (Extended Results) | `scripts/run_extensions.py` | λ-sweep, β3/β4 attack-strength sweeps, 3 failure cases, MLP backbone |
| Figures (incl. Pareto, λ-sweep, strength-sweep) | `scripts/plot_results.py` | Renders the six paper PDFs |

---

## Key Results (5 seeds: `{20260417, 1, 2, 3, 4}`, mean ± std)

### Table 1 — Main experiment across 5 threat scenarios

Target marginal coverage `1 − α = 0.90`; calibration is held fixed across threats. **CRIT cov.** is the CRITICAL-class conditional coverage (the safety-relevant quantity).

| Scenario | Coverage | CRIT cov. | Set size | Crit. err. (base) | Crit. err. (after abstain) | Abstain |
|---|---:|---:|---:|---:|---:|---:|
| **T0 Clean**            | 0.905±0.009 | 0.935±0.040 | 2.09±0.06 | 0.319±0.034 | **0.279±0.035** | 0.279±0.046 |
| **T1 KG corruption**    | 0.914±0.010 | 0.945±0.019 | 2.07±0.05 | 0.224±0.033 | **0.150±0.034** | 0.288±0.030 |
| **T2 Rule poisoning**   | 0.903±0.014 | 0.918±0.052 | 2.10±0.08 | 0.379±0.105 | **0.343±0.114** | 0.276±0.057 |
| **T3 Feature attack**   | 0.883±0.008 | 0.931±0.030 | 2.07±0.06 | 0.324±0.023 | **0.275±0.035** | 0.567±0.049 |
| **T4 Covariate shift**  | 0.686±0.006 | 0.743±0.026 | 1.99±0.06 | 0.461±0.027 | **0.289±0.054** | 0.628±0.028 |

Key takeaways reproduced in the paper:

- Marginal coverage stays within one empirical s.e. of the 90% target on T0–T3; CRITICAL-class coverage meets or exceeds 0.90 on all four exchangeable scenarios.
- Under T4 exchangeability is intentionally broken (marginal coverage drops to 0.686); the martingale detects the shift within 42±22 steps and the policy layer cuts the critical-class miss rate from 0.461 to **0.289**.
- T3 (feature attack) is detected within 275±149 steps; peak martingale 7.0×10¹⁰ with zero empirical false alarms on T0–T2 (Ville bound 0.05).

### Table 2 — Ablation under T4 covariate shift (5 seeds)

| Variant | Coverage | Avg set size | Crit. err. (after abstain) | Abstain rate |
|---|---:|---:|---:|---:|
| `no_conformal`  (set = all classes) | 1.000±0.000 | 4.00±0.00 | 0.331±0.054 | 0.486±0.007 |
| `no_martingale` (only set-size abstention) | 0.686±0.006 | 1.99±0.06 | 0.401±0.039 | 0.235±0.039 |
| `no_abstention` (raw argmax)        | 0.686±0.006 | 1.99±0.06 | 0.461±0.027 | 0.000±0.000 |
| **`full` SkyCert**                  | 0.686±0.006 | 1.99±0.06 | **0.289±0.054** | 0.628±0.028 |

The full SkyCert configuration dominates every ablation variant on the safety-critical metric (post-abstention critical-class miss rate).

### Table 3 — Baseline comparison under T4 (matched abstention, 5 seeds)

| Method | Crit. after abstain | Abstain | Assurance properties |
|---|---:|---:|---|
| MSP threshold | 0.300±0.034 | 0.628±0.028 | none |
| Entropy threshold | 0.259±0.039 | 0.628±0.028 | none |
| Conformal-only | 0.401±0.039 | 0.235±0.039 | coverage only |
| **full SkyCert** | **0.289±0.054** | 0.628±0.028 | **coverage + drift + audit** |

At matched abstention rates, entropy thresholding achieves a slightly lower mean miss rate, but provides no coverage guarantee, no sequential drift detection, and no audit trail — exactly the properties a regulator requires.

### Appendix: extension experiments (`outputs/extensions.json`)

- **λ-sweep** (hybrid score weight): miss rate stable in `λ_drift ∈ [1.0, 2.0]` (0.279–0.279), within 1.1 pp of the minimum over `[0.3, 3.0]`.
- **β4 sweep** (covariate shift): base miss rises from 0.344 (β4=0.2) to 0.552 (β4=1.0); post-abstention miss stays in [0.26, 0.30].
- **β3 sweep** (feature attack): martingale fires at β3 ≥ 0.15, flattening the post-abstention miss curve.
- **MLP backbone**: on `{clean, input_manip, dist_shift}` the MLP gives lower critical-miss rates (0.110/0.129/0.141) than the logistic scorer, confirming that SkyCert's guarantees are model-agnostic.
- **3 failure cases** extracted for the distribution_shift stream, illustrating ACCEPT (residual miss), ABSTAIN (absorbed miss), and ESCALATE (severe non-exchangeability).

---

## Core Concepts

### SkyCert pipeline at a glance

```
 ┌─────────────┐   softmax    ┌────────────────────┐   APS set       ┌──────────────────────┐
 │  Neural     ├──probs───────► ConformalRiskSet   ├──prediction set─►  AssurancePolicy     │
 │  scorer     │              │  (split calib.)    │                 │  (ACCEPT / ABSTAIN / │
 └─────▲───────┘              └─────────▲──────────┘                 │   ALERT / ESCALATE)  │
       │ logits                         │ q̂_α                        │                      │
 ┌─────┴───────┐                        │                            │                      │
 │  Symbolic   │◄─ rule trace ──┐       │                            │                      │
 │  rule engine│                │       │                            │                      │
 └─────────────┘                │       │                            │                      │
                                │       │                            │                      │
                      ┌─────────┴───────┴──────────┐                 │                      │
                      │ Hybrid nonconformity        │                 │                      │
                      │ (1−maxP) + L2 drift         ├──martingale────►                      │
                      └─────────────────────────────┘                 └──────────┬──────────┘
                                                                                 │
                                                                      ┌──────────▼──────────┐
                                                                      │  AuditLogger        │
                                                                      │  (JSONL per op.)    │
                                                                      └─────────────────────┘
```

### Audit artifact example (abridged)

```json
{
  "op_id": 17421,
  "timestamp": "2026-04-17T12:34:56.789Z",
  "features": [0.82, 0.67, 0.44, 0.12, ...],
  "rule_trace": [
    {"rule_id": "R0", "predicate": "feat[0] > 0.8", "delta": [0,0,0.6,1.0], "fired": true},
    {"rule_id": "R3", "predicate": "feat[3] < 0.2", "delta": [0,-0.2,-0.3,-0.2], "fired": true}
  ],
  "probs": [0.05, 0.18, 0.33, 0.44],
  "prediction_set": ["HIGH", "CRITICAL"],
  "nonconformity": 0.81,
  "martingale": 2.41e+4,
  "decision": "ESCALATE",
  "reason": "martingale>=threshold AND |set|>=max_set_fraction"
}
```

---

## Integration with `SkyNetUamPlatform`

```python
from skycert.pipeline import SkyCertPipeline
from skycert.config import SkyCertConfig

cfg = SkyCertConfig.load("configs/default.yaml")
skycert = SkyCertPipeline(cfg)

skycert.fit(X_train, y_train)          # trains the neuro-symbolic base
skycert.calibrate(X_calib, y_calib)    # split-conformal calibration
                                       # + warm-starts the martingale

for op_id, x in streaming_ops():
    decision = skycert.step(x, sample_id=op_id)
    if decision.verdict == "ESCALATE":
        route_to_human_operator(op_id, decision.audit_record)
```

Drop-in contract for the platform:

- **Input**: any `RiskScorer` (e.g. the neural model currently used by `SkyKg` / `SkyFlow`) + a `SymbolicRuleEngine` built from a UAM knowledge-graph slice.
- **Output**: an `AssuranceDecision` (`ACCEPT` / `ABSTAIN` / `ALERT` / `ESCALATE`) with an attached machine-readable audit record.
- **Contract**: `ESCALATE` decisions MUST be routed to a human operator or a higher-authority controller; the platform MUST NOT autonomously approve an escalated operation.

---

## Running Tests

```bash
cd modules/SkyCert
python -m pytest tests -v
```

The suite checks:

- `test_conformal.py` — marginal coverage is within `O(n⁻¹ᐟ²)` of `1 − α`, prediction sets always include the top-1 class when it meets the threshold.
- `test_martingale.py` — `E[M_t] ≈ 1` under H0 (exchangeability) on a synthetic clean stream; the martingale fires within the expected horizon under an injected distribution shift.
- `test_policy.py` — the full decision matrix (ACCEPT / ABSTAIN / ALERT / ESCALATE) for every combination of set-size branch and martingale branch.

All 9 tests run in <3 s on CPU.

---

## Reviewer FAQ

**Q. Why synthetic data?**
Real CAAC / FAA UAM operational data cannot currently be publicly redistributed. The synthetic generator in `skycert/data/synthetic.py` is fully documented (parameter distributions, rule structure, class prior) and seeded; the threat-injection module (`threats.py`) is also deterministic. All claims in the paper are recoverable from the seed-pinned configuration in `configs/default.yaml`.

**Q. Are the coverage guarantees honest under threat?**
Yes — the conformal guarantee holds whenever calibration and test data are exchangeable. Table 1 shows coverage staying within `±0.03` of `1 − α` on T0–T3 (exchangeability preserved). Under T4 exchangeability is *intentionally* broken; the guarantee necessarily degrades, and this is exactly what the martingale is designed to flag (which it does in 40 steps).

**Q. What happens if I disable the martingale?**
See the `no_martingale` row of the ablation. Coverage and set size are unchanged, but the post-abstention critical-class miss rate rises from **0.289±0.054** to **0.401±0.039** (5-seed mean±std), because the policy layer no longer receives the online distribution-shift signal.

**Q. Is there a GPU / API dependency?**
No. The base model is a NumPy multinomial logistic regression; the symbolic engine is a pure-Python rule evaluator; the assurance layer is algebraic. `run.sh` finishes in ~30 s on a single CPU core.

**Q. What about the paper source?**
By platform policy, paper PDF and TeX sources are not committed to the public repository. Only this open-source module and its default reviewer configuration are public.

---

## Citation

```bibtex
@inproceedings{liu2026skycert,
  title     = {SkyCert: Conformal and Martingale-Based Runtime Security
               Assurance for Neuro-Symbolic Risk Reasoning in Urban Air
               Mobility},
  author    = {Liu, Yushu and Wang, Longbiao and Du, Chenglin and Zhai, Haixiao},
  booktitle = {Proceedings of the 2026 IEEE International Conference on
               Cyber Security and Cloud Computing (CSCloud)},
  year      = {2026}
}
```

## License

This project is licensed under the Apache License 2.0.
