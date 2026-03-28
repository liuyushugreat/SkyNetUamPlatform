#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════
#  SkyFlow — One-Click Paper Reproduction
#  Paper:  "SkyFlow: Temporal Relational Graph Attention for
#           Real-Time UAV Conflict Detection"
#  Venue:  ACM MobiHoc 2026
#  Repo:   github.com/liuyushugreat/SkyNetUamPlatform/modules/SkyFlow
# ════════════════════════════════════════════════════════════════════
#
#  Usage:
#    bash run.sh              # Full reproduction (~14h on A100)
#    bash run.sh --quick      # Quick verification (~5min on CPU)
#
#  Expected output (full run on A100, Table 3 in the paper):
#    Method      CDR↑      FAR↓      F1↑    Latency↓
#    VO         0.6012    0.4231    0.5847     8.4 ms
#    LSTM-P     0.7856    0.1923    0.7724    23.7 ms
#    Tfm-P      0.8367    0.1547    0.8241    41.2 ms
#    STGCN      0.8512    0.1389    0.8384    52.8 ms
#    GAT-S      0.8794    0.1156    0.8673   124.6 ms
#    TR-GAT-NT  0.8891    0.1023    0.8782   139.1 ms
#    TR-GAT     0.9247    0.0734    0.9132   147.3 ms
# ════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if command -v python3 &>/dev/null; then
    PYTHON=${PYTHON:-python3}
elif command -v python &>/dev/null; then
    PYTHON=${PYTHON:-python}
else
    echo "[ERROR] Neither python3 nor python found in PATH."
    exit 1
fi
DEVICE=${DEVICE:-auto}

# ── Parse arguments ──
QUICK=false
SKIP_BASELINES=false
EXTRA_ARGS=""
for arg in "$@"; do
    case $arg in
        --quick)       QUICK=true ;;
        --skip-baselines) SKIP_BASELINES=true ;;
        --device=*)    DEVICE="${arg#*=}" ;;
        *)             EXTRA_ARGS="$EXTRA_ARGS $arg" ;;
    esac
done

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  SkyFlow — ACM MobiHoc 2026 Paper Reproduction             ║"
echo "║  Temporal Relational Graph Attention for Real-Time          ║"
echo "║  UAV Conflict Detection                                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Step 0: Environment check ──
echo "[Step 0] Checking environment..."

PY_VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "none")
if [ "$PY_VERSION" = "none" ]; then
    echo "[ERROR] Python not found. Please install Python 3.10+."
    echo "        Recommended: conda create -n skyflow python=3.10 -y"
    exit 1
fi
echo "  Python:  $PY_VERSION"

# Check if skyflow package is importable
if ! $PYTHON -c "import skyflow" 2>/dev/null; then
    echo "  [INFO] SkyFlow not installed. Installing now..."
    $PYTHON -m pip install -e ".[dev]" --quiet
fi

# Verify core imports
$PYTHON -c "
from skyflow.config import SkyFlowConfig
from skyflow.models.tr_gat import TRGAT
from skyflow.training.trainer import SkyFlowTrainer
print('  Imports: OK')
"

# Check device
ACTUAL_DEVICE=$($PYTHON -c "
import torch
if '$DEVICE' == 'auto':
    d = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    d = '$DEVICE'
print(d)
")
echo "  Device:  $ACTUAL_DEVICE"

# Show seed protocol
echo "  Seeds:   42, 123, 456, 789, 1024"

if [ "$QUICK" = "true" ]; then
    echo "  Mode:    QUICK (50 UAVs, 10 epochs, 2 seeds — for pipeline verification)"
else
    echo "  Mode:    FULL (500 UAVs, 150 epochs, 5 seeds — matches paper)"
fi
echo ""

# ── Step 1: Run unit tests ──
echo "[Step 1] Running unit tests..."
if $PYTHON -m pytest --version &>/dev/null; then
    $PYTHON -m pytest tests/ -q --tb=line 2>&1 | tail -5
else
    echo "  [WARN] pytest not found; installing dev dependencies..."
    $PYTHON -m pip install -e ".[dev]" --quiet
    $PYTHON -m pytest tests/ -q --tb=line 2>&1 | tail -5
fi
echo ""

# ── Step 2: Run the full reproduction pipeline ──
echo "[Step 2] Running full reproduction pipeline..."
echo "         (Data generation → TR-GAT training → Baselines → Scalability → Significance tests → Figures)"
echo ""

ARGS="--config configs/default.yaml --device $DEVICE"
if [ "$QUICK" = "true" ]; then
    ARGS="$ARGS --quick"
fi
if [ "$SKIP_BASELINES" = "true" ]; then
    ARGS="$ARGS --skip-baselines"
fi

$PYTHON scripts/reproduce_paper.py $ARGS $EXTRA_ARGS

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  REPRODUCTION FINISHED                                     ║"
echo "║                                                            ║"
echo "║  Output files:                                             ║"
echo "║    outputs/all_results.json        — Table 3 metrics       ║"
echo "║    outputs/multi_seed_results.json  — Per-seed breakdown   ║"
echo "║    outputs/scalability_results.json — Table 7 latency      ║"
echo "║    outputs/significance_tests.json  — Table 5 t-tests      ║"
echo "║    outputs/charts/                  — Publication figures   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
