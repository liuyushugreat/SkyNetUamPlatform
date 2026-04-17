# SkyFlow: Temporal Relational Graph Attention for Real-Time UAV Conflict Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Conference: MobiHoc 2026](https://img.shields.io/badge/Conference-MobiHoc_2026-blue.svg)](https://www.sigmobile.org/mobihoc/2026/)
[![Tests: 23 passed](https://img.shields.io/badge/tests-23_passed-brightgreen.svg)]()

> **Official implementation** of the SkyFlow framework and **TR-GAT** (Temporally-conditioned Relational Graph Attention Network), an edge-assisted real-time conflict detection and resolution system for dense Urban Air Mobility (UAM) and Flying Ad Hoc Networks (FANETs).

---

## Artifact Overview

This artifact provides the complete source code to reproduce all experimental results presented in the paper, including detection accuracy, system latency, baseline comparisons, scalability analysis, and statistical significance tests.

| Component | Description |
|-----------|-------------|
| **UrbanAir-500 Benchmark** | High-fidelity UAM simulator: 500 concurrent UAVs, 5 km x 5 km urban grid, stochastic wind, GPS noise (CEP 2.5 m), ADS-B latency (0.5--1.2 s) |
| **Temporal Knowledge Graph Builder** | Converts raw ADS-B telemetry, flight plans, weather grids, and corridor reservations into typed spatio-temporal knowledge graphs at 10 Hz |
| **TR-GAT Model** | 4-layer temporally-conditioned relational graph attention with sinusoidal temporal encoding and multi-relation gating |
| **Resolution Module** | Coordinated avoidance waypoint generation via Projected Gradient Descent (PGD) for conflict clusters up to 12 aircraft |
| **6 Baselines** | VO, LSTM-Pair, Transformer-Pair, STGCN, GAT-Static, TR-GAT-NoTemp -- all parameter-matched |

### Key Results (UrbanAir-500, 500 simultaneous UAVs)

| Method | CDR | FAR | F1 | Latency (ms) |
|--------|:---:|:---:|:--:|:------------:|
| Velocity Obstacle | 0.6012 | 0.4231 | 0.5847 | 8.4 |
| LSTM-Pair | 0.7856 | 0.1923 | 0.7724 | 23.7 |
| Transformer-Pair | 0.8367 | 0.1547 | 0.8241 | 41.2 |
| STGCN | 0.8512 | 0.1389 | 0.8384 | 52.8 |
| GAT-Static | 0.8794 | 0.1156 | 0.8673 | 124.6 |
| TR-GAT-NoTemp | 0.8891 | 0.1023 | 0.8782 | 139.1 |
| **TR-GAT (Ours)** | **0.9247** | **0.0734** | **0.9132** | **147.3** |

---

## Quick Installation

**Prerequisites:** Python 3.10+, PyTorch 2.2+. Tested on Ubuntu 22.04 with NVIDIA A100 (CUDA 12.1).

```bash
# Clone and navigate
git clone https://github.com/liuyushugreat/SkyNetUamPlatform.git
cd SkyNetUamPlatform/modules/SkyFlow

# Option A: pip editable install (recommended)
pip install -e ".[dev]"

# Option B: conda environment
conda create -n skyflow python=3.10 -y
conda activate skyflow
pip install -e ".[dev]"

# Verify installation (runs 23 unit tests)
bash scripts/setup_env.sh
```

---

## 1-Click Reproducibility (For Reviewers)

```bash
# Full reproduction — all tables and figures (~14h on A100)
bash run.sh

# Quick pipeline verification (~5 min on CPU)
bash run.sh --quick
```

This single command installs dependencies, runs 23 unit tests, trains TR-GAT across 5 seeds, evaluates all 6 baselines, performs the scalability sweep, runs Bonferroni-corrected significance tests, and generates all publication figures.

### Individual Tables

```bash
bash scripts/reproduce_table3.sh      # Table 3: Detection performance
bash scripts/reproduce_table7.sh      # Table 7: Scalability analysis
```

---

## Training from Scratch

```bash
# Full reproduction: data generation + TR-GAT (5 seeds) + baselines + figures
python scripts/reproduce_paper.py --config configs/default.yaml

# Train TR-GAT only
python scripts/train.py --config configs/default.yaml

# Evaluate a checkpoint
python scripts/evaluate.py --checkpoint outputs/best_model.pt

# Run all baselines
python scripts/run_baselines.py

# Scalability sweep (Table 7)
python scripts/eval_scalability.py

# SDD zero-shot transfer (Table 6, requires SDD data)
python scripts/eval_sdd_transfer.py --sdd-root /path/to/SDD
```

### Hyperparameters

All hyperparameters are specified in `configs/default.yaml` and match Table 2 in the paper:

| Parameter | Value |
|-----------|-------|
| TR-GAT layers | 4 |
| Embedding dim | 128 |
| Attention heads | 4 |
| Temporal encoding dim | 32 |
| Recurrent state dim | 64 |
| Observation window K | 10 epochs (1 s) |
| Conflict threshold | 0.42 |
| Focal loss gamma | 2.0 |
| Learning rate | 3e-4 (AdamW) |
| Training epochs | 150 |
| Seeds | 42, 123, 456, 789, 1024 |

---

## Architecture

```
ADS-B Telemetry ─┐
Flight Plans ────┤                         ┌──────────────┐
Weather Grid ────┼──► TKG Builder (Alg.1) ─┤  TR-GAT      ├──► Conflict Head ──► PGD Resolution
Corridor Log ────┘    (< 12 ms)            │  L=4 layers  │    (Eq. 6, 2-MLP)   (Alg. 3, 20 steps)
                                           │  K=10 window  │
                   6 relation types        │  GRU recur.   │    Pairwise p_ij
                   + temporal encoding     └──────────────┘    + avoidance Δp_i
                     φ(δ) (Eq. 2)
```

---

## Repository Structure

```
SkyFlow/
├── run.sh                         # ← START HERE: one-click reproduction
├── skyflow/                       # Core Python package
│   ├── models/
│   │   ├── tr_gat.py              # TR-GAT: Eq. (3)-(5), Algorithm 2
│   │   ├── temporal_encoding.py   # Sinusoidal temporal encoding φ(δ), Eq. (2)
│   │   ├── conflict_head.py       # Pairwise conflict scoring, Eq. (6)
│   │   └── resolution.py          # PGD resolution solver, Algorithm 3
│   ├── data/
│   │   ├── tkg_builder.py         # TKG construction, Algorithm 1
│   │   ├── urbanair500.py         # UrbanAir-500 benchmark (Table 1)
│   │   └── sdd_adapter.py         # Stanford Drone Dataset adapter
│   ├── baselines/
│   │   ├── velocity_obstacle.py   # Reciprocal Velocity Obstacles
│   │   ├── lstm_pair.py           # LSTM-Pair encoder
│   │   ├── transformer_pair.py    # Transformer-Pair encoder
│   │   ├── stgcn.py               # Spatio-Temporal GCN
│   │   └── gat_static.py          # Static GAT (no temporal encoding)
│   ├── training/
│   │   ├── trainer.py             # AdamW + warmup + cosine, K=10 windows
│   │   ├── losses.py              # Focal loss (γ=2, α=0.75)
│   │   └── metrics.py             # CDR, FAR, F1, regime analysis, t-tests
│   └── utils/
│       └── visualization.py       # Publication-quality figures
├── scripts/
│   ├── reproduce_paper.py         # One-click: all tables + figures
│   ├── reproduce_table3.sh        # Bash: Table 3 (detection performance)
│   ├── reproduce_table7.sh        # Bash: Table 7 (scalability)
│   ├── setup_env.sh               # Environment setup + verification
│   ├── train.py                   # Training entry point
│   ├── evaluate.py                # Evaluation entry point
│   ├── run_baselines.py           # Baseline comparison
│   ├── eval_scalability.py        # Fleet-size latency sweep
│   └── eval_sdd_transfer.py       # SDD zero-shot transfer
├── configs/
│   └── default.yaml               # All hyperparameters (Table 2)
├── tests/                         # 23 unit tests
│   ├── test_tr_gat.py
│   ├── test_tkg_builder.py
│   └── test_metrics.py
├── pyproject.toml                 # Package metadata & dependencies
└── requirements.txt               # Pinned dependencies (alternative)
```
---
## License

This project is licensed under the Apache License 2.0.
