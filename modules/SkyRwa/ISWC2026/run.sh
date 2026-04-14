#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# SkyRwa — ISWC 2026 One-Click Reproduction Script
# Paper: "Modeling Governable Flight-to-Asset Lifecycles
#         with Knowledge Graphs, SHACL, and Provenance"
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

echo ""
echo "================================================"
echo "  SkyRwa ISWC 2026 Artifact Reproduction"
echo "================================================"
echo ""

# ── Step 1: Generate benchmark dataset (Table 5) ──
echo "[Step 1/8] Generating benchmark dataset (Table 5: 105 flights, 10 scenarios)..."
$PYTHON reproduce_table5.py
echo ""

# ── Step 2: Baseline comparison (Table 6) ──
echo "[Step 2/8] Running baseline comparison: JSON-scan vs SPARQL (Table 6)..."
$PYTHON reproduce_table6.py
echo ""

# ── Step 3: Governance ablation (Table 7) ──
echo "[Step 3/8] Running governance ablation: Python vs SHACL vs Combined (Table 7)..."
$PYTHON reproduce_table7.py
echo ""

# ── Step 4: Scalability (Table 8) ──
echo "[Step 4/8] Running scalability experiment: 5–1000 flights (Table 8)..."
$PYTHON reproduce_table8.py
echo ""

# ── Step 5: SPARQL competency questions (Table 9) ──
echo "[Step 5/8] Running SPARQL competency questions (Table 9)..."
$PYTHON reproduce_table9.py
echo ""

# ── Step 6: Robustness (multi-run, scale, thresholds) ──
echo "[Step 6/8] Running robustness experiments..."
$PYTHON reproduce_robustness.py
echo ""

# ── Step 7: Case studies (Section 7.7) ──
echo "[Step 7/8] Running case studies (Section 7.7)..."
$PYTHON reproduce_case_studies.py
echo ""

# ── Step 8: SHACL + governance validation ──
echo "[Step 8/8] Running SHACL + governance validation..."
$PYTHON reproduce_validation.py
echo ""

echo "================================================"
echo "  All steps completed successfully."
echo "  JSON results: outputs/"
echo "  Benchmark data: data/"
echo "================================================"
