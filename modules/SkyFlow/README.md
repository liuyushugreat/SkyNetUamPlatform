# SkyFlow: Temporal Knowledge Graph Reasoning for Multi-UAV Conflict Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/pytorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A module of [SkyNetUamPlatform](https://github.com/liuyushugreat/SkyNetUamPlatform) implementing
**TR-GAT** (Temporal Relational Graph Attention Network) for real-time multi-UAV conflict detection
in dense low-altitude urban airspace.

## Overview

SkyFlow models airspace dynamics as an evolving **Temporal Knowledge Graph (TKG)** and performs
anticipatory conflict detection through multi-hop relational reasoning over UAV trajectories,
flight intents, and environmental constraints.

### Key Results (UrbanAir-500 Benchmark, 500 simultaneous UAVs)

| Metric | Value |
|--------|-------|
| Conflict Detection Rate (CDR) | 92.47% |
| False Alert Rate (FAR) | 7.34% |
| F1 Score | 0.9132 |
| 95th-pctl Latency | 147.3 ms |
| Parameters | 4.2M |

## Architecture

```
ADS-B Telemetry ─┐
Flight Plans ────┤
Weather Grid ────┼──► TKG Builder ──► TR-GAT (L=4) ──► Conflict Head ──► Resolution
Corridor Log ────┘       │                  │                │
                   Typed Nodes         Temporal           Pairwise
                   & Edges          Attention +         Probabilities
                                  Relation Gating
```

### Components

- **TKG Builder**: Constructs temporal knowledge graphs from multi-source airspace telemetry at 10 Hz
- **TR-GAT**: Stack of 4 temporally-conditioned relational graph attention layers with sinusoidal temporal encoding
- **Conflict Scoring Head**: 2-layer MLP producing pairwise conflict probabilities
- **Resolution Module**: Coordinated avoidance waypoint generation via projected gradient descent

## Quick Start

### Installation

```bash
cd modules/SkyFlow
pip install -e ".[dev]"
```

### Reproduce Paper Results

```bash
# Full reproduction (training + evaluation + all baselines)
python scripts/reproduce_paper.py

# Train TR-GAT only
python scripts/train.py --config configs/default.yaml

# Evaluate a trained checkpoint
python scripts/evaluate.py --checkpoint outputs/best_model.pt

# Run all baselines
python scripts/run_baselines.py
```

### Quick Verification (CPU, ~5 min)

```bash
python scripts/reproduce_paper.py --quick --device cpu
```

## Project Structure

```
SkyFlow/
├── skyflow/
│   ├── models/
│   │   ├── tr_gat.py              # TR-GAT architecture
│   │   ├── temporal_encoding.py   # Sinusoidal temporal encoding φ(δ)
│   │   ├── conflict_head.py       # Pairwise conflict scoring MLP
│   │   └── resolution.py          # Coordinated resolution module
│   ├── data/
│   │   ├── tkg_builder.py         # Temporal Knowledge Graph construction
│   │   ├── urbanair500.py         # UrbanAir-500 benchmark simulator
│   │   └── sdd_adapter.py         # Stanford Drone Dataset adapter
│   ├── baselines/
│   │   ├── velocity_obstacle.py   # Reciprocal Velocity Obstacles
│   │   ├── lstm_pair.py           # LSTM-Pair encoder
│   │   ├── transformer_pair.py    # Transformer-Pair encoder
│   │   ├── stgcn.py               # Spatio-Temporal GCN
│   │   └── gat_static.py         # Static GAT (no temporal encoding)
│   ├── training/
│   │   ├── trainer.py             # Training loop with cosine annealing
│   │   ├── losses.py              # Focal loss with class reweighting
│   │   └── metrics.py             # CDR, FAR, F1, latency measurement
│   └── utils/
│       └── visualization.py       # Result charts and figures
├── scripts/
│   ├── train.py                   # Training entry point
│   ├── evaluate.py                # Evaluation entry point
│   ├── run_baselines.py           # Baseline comparison
│   └── reproduce_paper.py         # One-click paper reproduction
├── configs/
│   └── default.yaml               # Default hyperparameters
└── tests/
    ├── test_tr_gat.py
    ├── test_tkg_builder.py
    └── test_metrics.py
```

## Citation

If you use SkyFlow in your research, please cite:

```bibtex
@inproceedings{skyflow2026,
  title={SkyFlow: Temporal Knowledge Graph Reasoning with Graph Neural Networks
         for Real-Time Multi-UAV Conflict Detection in Low-Altitude Airspace},
  author={Anonymous},
  booktitle={Proceedings of ACM MobiHoc},
  year={2026}
}
```

## License

This project is part of [SkyNetUamPlatform](https://github.com/liuyushugreat/SkyNetUamPlatform)
and is released under the MIT License.
