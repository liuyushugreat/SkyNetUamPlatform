#!/usr/bin/env bash
# ============================================================
# Reproduce Table 3 — Overall Detection Performance & Latency
# Paper: "SkyFlow: Temporal Relational Graph Attention for
#         Real-Time UAV Conflict Detection"
# Conference: ACM MobiHoc 2026
#
# This script trains TR-GAT and all 6 baselines across 5 seeds,
# then prints the comparison table matching Table 3 in the paper.
#
# Expected output (full run on A100):
#   Method      CDR↑     FAR↓      F1↑    Latency(ms)↓
#   VO         0.6012   0.4231   0.5847        8.4
#   LSTM-P     0.7856   0.1923   0.7724       23.7
#   Tfm-P      0.8367   0.1547   0.8241       41.2
#   STGCN      0.8512   0.1389   0.8384       52.8
#   GAT-S      0.8794   0.1156   0.8673      124.6
#   TR-GAT-NT  0.8891   0.1023   0.8782      139.1
#   TR-GAT     0.9247   0.0734   0.9132      147.3
#
# Estimated runtime: ~14 hours on A100, ~5 min with --quick
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if command -v python3 &>/dev/null; then
    PYTHON=${PYTHON:-python3}
elif command -v python &>/dev/null; then
    PYTHON=${PYTHON:-python}
else
    echo "[ERROR] Neither python3 nor python found in PATH."
    exit 1
fi
DEVICE=${DEVICE:-auto}
QUICK=${QUICK:-false}

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Reproducing Table 3: Overall Detection Performance     ║"
echo "║  SkyFlow — ACM MobiHoc 2026                             ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Device:  $DEVICE"
echo "  Config:  configs/default.yaml"
echo "  Seeds:   42, 123, 456, 789, 1024"
echo ""

ARGS="--config configs/default.yaml --device $DEVICE"
if [ "$QUICK" = "true" ]; then
    echo "  [!] Quick mode enabled (reduced UAVs & epochs)"
    ARGS="$ARGS --quick"
fi

echo "Starting full pipeline (data → training → baselines → figures)..."
echo ""

$PYTHON scripts/reproduce_paper.py $ARGS

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Table 3 reproduction complete."
echo "  Results saved to: outputs/all_results.json"
echo "  Figures saved to: outputs/charts/"
echo "════════════════════════════════════════════════════════════"
