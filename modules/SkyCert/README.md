# SkyCert: Uncertainty-Calibrated Neuro-Symbolic Reasoning with Conformal Prediction for Certifiable Risk Assessment in Urban Air Mobility

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Conference: ESORICS 2026](https://img.shields.io/badge/Conference-ESORICS_2026-green.svg)](https://esorics2026.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Runtime: CPU-only](https://img.shields.io/badge/runtime-CPU--only-lightgrey.svg)]()
[![Reproducible: seed 20260417](https://img.shields.io/badge/seed-20260417-informational)]()

> **ESORICS 2026 Reviewers — one-click reproduction:**
> ```bash
> cd modules/SkyCert
> bash run.sh          # Linux/macOS (≈30 s, CPU-only)
> # .\run.ps1          # Windows PowerShell
> ```
> The script installs dependencies, runs 9 unit tests, reproduces the 5-threat main experiment and the 4-variant ablation, renders the three paper figures (`outputs/figs/`), and prints a summary table with the numbers cited in the paper. No GPU, API key, or network access is required.

---

## Overview

**SkyCert** is the *assurance layer* of `SkyNetUamPlatform`. It wraps an existing neuro-symbolic risk reasoner (neural scorer + symbolic rule engine over a UAM knowledge-graph slice) with three complementary mechanisms that turn opaque risk scores into **auditable, certification-friendly decisions**:

1. **Conformal Risk Sets** — split-conformal prediction with an Adaptive Prediction Sets (APS) nonconformity score; converts raw softmax outputs into prediction sets with a finite-sample marginal-coverage guarantee (`1 − α`).
2. **Martingale Monitoring** — an online test-martingale over a *hybrid* nonconformity stream (confidence slack + standardized input drift) driven by a simple-jumper betting function; provides anytime-valid sequential evidence against the exchangeability hypothesis.
3. **Abstention / Alert / Escalation Policy** — a decision-gating layer that refuses to emit a confident answer when (i) the risk set is uninformative, (ii) the martingale breaches a pre-registered threshold, or (iii) the rule trace disagrees with the neural prediction.

Every decision is persisted as a **machine-readable audit artifact** (JSON Lines) containing the input operation ID, the symbolic rule trace, the conformal prediction set, the martingale trajectory, and the final policy verdict. These artifacts are designed to back an offline certification argument for UAM operational authorizations.

This module is the companion artifact for the ESORICS 2026 submission:

> *SkyCert: Uncertainty-Calibrated Neuro-Symbolic Reasoning with Conformal Prediction for Certifiable Risk Assessment in Urban Air Mobility.*

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
├── pyproject.toml            ← Package metadata (skycert, MIT/Apache-compatible)
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
├── scripts/                  ← Experiment entry points (all CPU-only, <30 s total)
│   ├── run_experiment.py     ←   Main: 5 threat scenarios → metrics.json + audit/
│   ├── run_ablation.py       ←   Ablation: 4 variants under T4 → ablation.json
│   └── plot_results.py       ←   Renders 3 paper figures to outputs/figs/
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
| `outputs/metrics.json` | 5 threat scenarios × {coverage, avg set size, ECE, top-1 accuracy, critical-error rate before/after abstention, abstain/alert/escalation rates, martingale max, detection delay, false-alarm rate} |
| `outputs/ablation.json` | 4 ablation variants under the covariate-shift threat |
| `outputs/audit/audit_<scenario>.jsonl` | Per-decision audit artifacts (1 JSON object per UAM operation) |
| `outputs/audit_ablation/<variant>.jsonl` | Per-decision audit artifacts for each ablation variant |
| `outputs/figs/coverage_vs_threat.pdf` | Figure 3 — empirical coverage and ECE under every threat |
| `outputs/figs/critical_error.pdf`     | Figure 4 — critical-class miss rate before vs. after abstention |
| `outputs/figs/martingale_max.pdf`     | Figure 5 — max martingale value per threat (log scale) |

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

# 5. paper figures              →  outputs/figs/*.pdf
python -m scripts.plot_results   --config configs/default.yaml
```

All randomness is seeded from `configs/default.yaml` (`seed: 20260417`); re-running on the same Python/NumPy version yields bit-identical `metrics.json`.

---

## Paper-to-Code Mapping

| Paper section | Code / script | What it reproduces |
|---------------|---------------|--------------------|
| §3 Threat Model (T1–T4) | `skycert/data/threats.py` | Injection procedures for KG corruption, rule poisoning, feature attack, covariate shift |
| §4.1 Neuro-Symbolic Base | `skycert/base/{neural,symbolic,neuro_symbolic}.py` | Class-balanced neural scorer + rule engine with audit trace |
| §4.2 Conformal Risk Sets | `skycert/assurance/conformal.py` | APS & LAC nonconformity, split calibration, coverage guarantee |
| §4.3 Martingale Monitor | `skycert/assurance/martingale.py` | Simple-jumper test-martingale with warm-start from calibration |
| §4.4 Hybrid Nonconformity | `skycert/pipeline.py::_nonconformity` | `(1 − max_prob) + L2-drift` in standardized feature space |
| §4.5 Decision Policy | `skycert/assurance/policy.py` | ACCEPT / ABSTAIN / ALERT / ESCALATE matrix |
| §4.6 Audit Artifacts | `skycert/assurance/audit.py` | JSONL per-decision records (inputs, rule trace, set, martingale) |
| §5 Implementation | `pyproject.toml`, `requirements.txt` | Pinned deterministic environment |
| §6.1 Main Experiment (Table 1, Fig. 3–5) | `scripts/run_experiment.py` | 5 threat scenarios × full metric panel |
| §6.2 Ablation (Table 2) | `scripts/run_ablation.py` | 4 variants: `no_conformal`, `no_martingale`, `no_abstention`, `full` |
| §6.3 Figures | `scripts/plot_results.py` | Renders the three PDF figures |

---

## Key Results (reproduced by `run.sh` on seed `20260417`)

### Table 1 — Main experiment across 5 threat scenarios

Target marginal coverage `1 − α = 0.90`; calibration is held fixed across threats.

| Scenario | Coverage | Set size | ECE | Top-1 | Crit. err. (base) | Crit. err. (after abstain) | Abstain | Alert | Escal. | M_max | Delay |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **T0 Clean**            | 0.899 | 2.08 | 0.124 | 0.617 | 0.364 | **0.312** | 0.269 | 0.000 | 0.000 | 7.10      | — |
| **T1 KG corruption**    | 0.905 | 2.07 | 0.119 | 0.615 | 0.240 | **0.179** | 0.276 | 0.000 | 0.000 | 4.69      | — |
| **T2 Rule poisoning**   | 0.909 | 2.27 | 0.092 | 0.585 | 0.403 | **0.348** | 0.405 | 0.000 | 0.000 | 6.53      | — |
| **T3 Feature attack**   | 0.881 | 2.06 | 0.097 | 0.589 | 0.351 | **0.267** | 0.528 | 0.358 | 0.097 | 2.2 × 10⁸ | 413 |
| **T4 Covariate shift**  | 0.691 | 1.97 | 0.076 | 0.442 | 0.494 | **0.290** | 0.623 | 0.486 | 0.093 | 1.3 × 10⁵⁵| 40 |

Key takeaways reported in the paper:

- Empirical coverage stays within `±0.03` of the 90% target on T0–T3 (conformal guarantee honored even under KG corruption and rule poisoning).
- Under T4 the i.i.d. assumption is *visibly* broken (coverage drops to 0.691) — but the martingale detects the shift within **40 steps** and the policy layer **halves** the critical-class miss rate (0.494 → 0.290) via abstention/alerting.
- Under T3 the martingale still reaches `2.2 × 10⁸` (far above the registered threshold of 20) with zero false alarms on T0–T2.

### Table 2 — Ablation under T4 covariate shift

| Variant | Coverage | Avg set size | Crit. err. (after abstain) | Abstain rate |
|---|---:|---:|---:|---:|
| `no_conformal`  (set = all classes) | 1.000 | 4.00 | 0.320 | 0.483 |
| `no_martingale` (only set-size abstention) | 0.679 | 1.96 | 0.329 | 0.542 |
| `no_abstention` (raw argmax)        | 0.679 | 1.96 | 0.376 | 0.386 |
| **`full` SkyCert**                  | 0.679 | 1.96 | **0.275** | 0.620 |

The full SkyCert configuration dominates every ablation variant on the safety-critical metric (post-abstention critical-class miss rate).

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
See the `no_martingale` row of the ablation. Coverage and set size are unaffected, but the critical-class miss rate rises from **0.275** to **0.329**, because the policy layer no longer receives the online distribution-shift signal.

**Q. Is there a GPU / API dependency?**
No. The base model is a NumPy multinomial logistic regression; the symbolic engine is a pure-Python rule evaluator; the assurance layer is algebraic. `run.sh` finishes in ~30 s on a single CPU core.

**Q. What about the paper source?**
By platform policy, paper PDF and TeX sources are not committed to the public repository. Only this open-source module and its default reviewer configuration are public.

---

## Citation

```bibtex
@inproceedings{liu2026skycert,
  title     = {SkyCert: Uncertainty-Calibrated Neuro-Symbolic Reasoning with
               Conformal Prediction for Certifiable Risk Assessment in
               Urban Air Mobility},
  author    = {Liu, Yushu and Wang, Longbiao and Du, Chenglin and Zhai, Haixiao},
  booktitle = {Proceedings of the 31st European Symposium on Research in
               Computer Security (ESORICS)},
  year      = {2026}
}
```

## License

This project is licensed under the Apache License 2.0.
