#!/usr/bin/env bash
set -euo pipefail

# ============================================================
#  SkyKG — KSEM 2026 One-Click Reproduction Script
#  Paper: "SkyKG: A Neuro-Symbolic Knowledge Graph Framework
#          for Explainable Risk Reasoning in Urban Air Mobility"
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Detect Python ──
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "[ERROR] Python not found. Please install Python 3.10+."
    exit 1
fi
echo "[INFO] Using Python: $($PYTHON --version)"

# ── Install dependencies ──
if [ -f requirements.txt ]; then
    echo "[Step 0] Installing dependencies..."
    $PYTHON -m pip install -r requirements.txt --quiet
fi

# ── Check DeepSeek API key ──
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
    if [ -f "$PROJECT_ROOT/.env" ]; then
        echo "[INFO] Loading .env from project root"
    else
        echo ""
        echo "[WARN] DEEPSEEK_API_KEY not set and no .env found."
        echo "       LLM-based methods (Direct LLM, SkyKG) will fail."
        echo "       Set it via:  export DEEPSEEK_API_KEY=your_key"
        echo ""
    fi
fi

echo ""
echo "========================================"
echo "  SkyKG KSEM 2026 Artifact Reproduction"
echo "========================================"
echo ""

# ── Step 1: Generate synthetic dataset (if not present) ──
if [ ! -f data/ksem_large_dataset.json ]; then
    echo "[Step 1] Generating 1,000-case synthetic dataset..."
    $PYTHON generate_large_scale_dataset.py
else
    echo "[Step 1] Dataset already exists (data/ksem_large_dataset.json), skipping."
fi

# ── Step 2: Run main benchmark (Table 2) ──
echo ""
echo "[Step 2] Running benchmark: Rule-Based vs Direct LLM vs SkyKG (Table 2)..."
$PYTHON benchmark_comparison.py

# ── Step 3: Run explanation quality evaluation (Table 4) ──
echo ""
echo "[Step 3] Running explanation quality evaluation (Table 4)..."
$PYTHON reproduce_table4.py

# ── Step 4: Run latency analysis (Fig. 5) ──
echo ""
echo "[Step 4] Running latency analysis (Fig. 5)..."
$PYTHON analyze_latency_tradeoff.py

# ── Step 5: Generate figures ──
echo ""
echo "[Step 5] Generating paper figures..."
$PYTHON viz_ontology_structure.py
$PYTHON viz_arch_placeholder.py

echo ""
echo "========================================"
echo "  All steps completed successfully."
echo "  Results:  outputs/"
echo "========================================"
