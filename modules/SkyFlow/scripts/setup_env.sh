#!/usr/bin/env bash
# ============================================================
# SkyFlow Environment Setup
# Installs all dependencies for reproducing the paper results.
# Tested on: Ubuntu 22.04 / macOS 14 / Windows 11 (WSL2)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "============================================================"
echo "  SkyFlow — Environment Setup"
echo "  Project root: $PROJECT_ROOT"
echo "============================================================"
echo ""

# Check Python version
if command -v python3 &>/dev/null; then
    PYTHON=${PYTHON:-python3}
elif command -v python &>/dev/null; then
    PYTHON=${PYTHON:-python}
else
    echo "[ERROR] Neither python3 nor python found in PATH."
    exit 1
fi
PY_VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$($PYTHON -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$($PYTHON -c 'import sys; print(sys.version_info.minor)')

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    echo "[ERROR] Python 3.10+ is required (found $PY_VERSION)."
    exit 1
fi
echo "[OK] Python $PY_VERSION detected."

# Install in editable mode with dev dependencies
echo ""
echo "[1/3] Installing SkyFlow package (editable mode)..."
cd "$PROJECT_ROOT"
$PYTHON -m pip install -e ".[dev]" --quiet

# Verify core imports
echo "[2/3] Verifying core imports..."
$PYTHON -c "
from skyflow.config import SkyFlowConfig
from skyflow.models.tr_gat import TRGAT
from skyflow.models.conflict_head import ConflictScoringHead
from skyflow.models.resolution import ResolutionModule
from skyflow.data.tkg_builder import TKGBuilder
from skyflow.data.urbanair500 import UrbanAir500
from skyflow.training.trainer import SkyFlowTrainer
from skyflow.training.metrics import ConflictMetrics
print('[OK] All core modules imported successfully.')
"

# Run unit tests
echo "[3/3] Running unit tests..."
$PYTHON -m pytest tests/ -q --tb=short

echo ""
echo "============================================================"
echo "  Setup complete. You can now run:"
echo ""
echo "    # Full paper reproduction"
echo "    python scripts/reproduce_paper.py"
echo ""
echo "    # Quick verification (~5 min on CPU)"
echo "    python scripts/reproduce_paper.py --quick --device cpu"
echo "============================================================"
