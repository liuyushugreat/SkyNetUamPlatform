#!/usr/bin/env bash
# ============================================================
# Reproduce Table 7 — Scalability & Latency Analysis
# Paper: "SkyFlow: Temporal Relational Graph Attention for
#         Real-Time UAV Conflict Detection"
# Conference: ACM MobiHoc 2026
#
# Sweeps fleet sizes [100, 200, 300, 400, 500] and measures
# 95th-percentile latency breakdown:
#   - Graph construction (TKG Builder)
#   - TR-GAT forward pass
#   - Total end-to-end
#
# Expected output (on A100):
#   UAVs    Graph(ms)  TR-GAT(ms)  Total(ms)
#    100        3.1       12.4       15.5
#    200        7.8       24.7       32.5
#    300       16.2       41.3       57.5
#    400       32.4       62.8       95.2
#    500       58.2       89.1      147.3
#
# Estimated runtime: ~30 min on A100
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
CHECKPOINT=${CHECKPOINT:-outputs/best_model.pt}

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Reproducing Table 7: Scalability & Latency Analysis    ║"
echo "║  SkyFlow — ACM MobiHoc 2026                             ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Device:      $DEVICE"
echo "  Checkpoint:  $CHECKPOINT"
echo "  Fleet sizes: 100, 200, 300, 400, 500"
echo ""

if [ ! -f "$CHECKPOINT" ]; then
    echo "[INFO] No checkpoint found at $CHECKPOINT."
    echo "       Running with random weights for latency measurement."
    echo "       (To use trained weights, run reproduce_table3.sh first.)"
    echo ""
fi

echo "Starting scalability sweep..."
echo ""

$PYTHON scripts/eval_scalability.py \
    --config configs/default.yaml \
    --checkpoint "$CHECKPOINT" \
    --device "$DEVICE" \
    --n-epochs 1000

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Table 7 reproduction complete."
echo "  Results saved to: outputs/scalability_results.json"
echo "════════════════════════════════════════════════════════════"
