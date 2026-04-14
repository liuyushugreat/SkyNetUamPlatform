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
echo "[Step 1/11] Generating benchmark dataset (Table 5: 105 flights, 10 scenarios)..."
$PYTHON reproduce_table5.py
echo ""

# ── Step 2: Baseline comparison (Table 6) ──
echo "[Step 2/11] Running baseline comparison: JSON-scan vs SPARQL (Table 6)..."
$PYTHON reproduce_table6.py
echo ""

# ── Step 3: Semantic baseline (Lifecycle KG vs Flat KG) ──
echo "[Step 3/11] Running semantic baseline: Lifecycle KG vs Flat KG..."
$PYTHON reproduce_semantic_baseline.py
echo ""

# ── Step 4: Governance ablation (Table 7) ──
echo "[Step 4/11] Running governance ablation: Python vs SHACL vs Combined (Table 7)..."
$PYTHON reproduce_table7.py
echo ""

# ── Step 5: Scalability (Table 8) ──
echo "[Step 5/11] Running scalability experiment: 5–1000 flights (Table 8)..."
$PYTHON reproduce_table8.py
echo ""

# ── Step 6: SPARQL competency questions (Table 9) ──
echo "[Step 6/11] Running SPARQL competency questions (Table 9)..."
$PYTHON reproduce_table9.py
echo ""

# ── Step 7: Robustness (multi-run, scale, thresholds) ──
echo "[Step 7/11] Running robustness experiments..."
$PYTHON reproduce_robustness.py
echo ""

# ── Step 8: Case studies ──
echo "[Step 8/11] Running case studies..."
$PYTHON reproduce_case_studies.py
echo ""

# ── Step 9: SHACL + governance validation ──
echo "[Step 9/11] Running SHACL + governance validation..."
$PYTHON reproduce_validation.py
echo ""

# ── Step 10: Ontology quality assessment ──
echo "[Step 10/11] Running ontology quality assessment..."
$PYTHON reproduce_ontology_quality.py
echo ""

# ── Step 11: Pilot expert evaluation ──
echo "[Step 11/11] Running pilot expert evaluation..."
$PYTHON reproduce_user_study.py
echo ""

echo "================================================"
echo "  All steps completed successfully."
echo "  JSON results: outputs/"
echo "  Benchmark data: data/"
echo "================================================"
