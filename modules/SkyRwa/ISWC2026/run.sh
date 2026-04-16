#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# SkyRwa 鈥?ISWC 2026 One-Click Reproduction Script
# Paper: "Modeling Governable Flight-to-Asset Lifecycles
#         with Knowledge Graphs, SHACL, and Provenance"
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 鈹€鈹€ Detect Python 鈹€鈹€
if command -v python3 &>/dev/null; then
  PYTHON=python3
elif command -v python &>/dev/null; then
  PYTHON=python
else
  echo "[ERROR] Python not found. Please install Python 3.10+."
  exit 1
fi
echo "[INFO] Using Python: $($PYTHON --version)"

# 鈹€鈹€ Install dependencies 鈹€鈹€
if [ -f requirements.txt ]; then
  echo "[Step 0] Installing dependencies..."
  $PYTHON -m pip install -r requirements.txt --quiet
fi

echo ""
echo "================================================"
echo "  SkyRwa ISWC 2026 Artifact Reproduction"
echo "================================================"
echo ""

# 鈹€鈹€ Step 1: Generate benchmark dataset (Table 5) 鈹€鈹€
echo "[Step 1/12] Generating benchmark dataset (Table 5: 105 flights, 10 scenarios)..."
$PYTHON reproduce_table5.py
echo ""

# 鈹€鈹€ Step 2: Baseline comparison (Table 6) 鈹€鈹€
echo "[Step 2/12] Running baseline comparison: JSON-scan vs SPARQL (Table 6)..."
$PYTHON reproduce_table6.py
echo ""

# 鈹€鈹€ Step 3: Semantic baseline (Lifecycle KG vs Flat KG) 鈹€鈹€
echo "[Step 3/12] Running semantic baseline: Lifecycle KG vs Flat KG..."
$PYTHON reproduce_semantic_baseline.py
echo ""

# 鈹€鈹€ Step 4: Governance ablation (Table 7) 鈹€鈹€
echo "[Step 4/12] Running governance ablation: Python vs SHACL vs Combined (Table 7)..."
$PYTHON reproduce_table7.py
echo ""

# 鈹€鈹€ Step 5: Scalability (Table 8) 鈹€鈹€
echo "[Step 5/12] Running scalability experiment: 5鈥?000 flights (Table 8)..."
$PYTHON reproduce_table8.py
echo ""

# 鈹€鈹€ Step 6: SPARQL competency questions (Table 9) 鈹€鈹€
echo "[Step 6/12] Running SPARQL competency questions (Table 9)..."
$PYTHON reproduce_table9.py
echo ""

# 鈹€鈹€ Step 7: Robustness (multi-run, scale, thresholds) 鈹€鈹€
echo "[Step 7/12] Running robustness experiments..."
$PYTHON reproduce_robustness.py
echo ""

# 鈹€鈹€ Step 8: Case studies 鈹€鈹€
echo "[Step 8/12] Running case studies..."
$PYTHON reproduce_case_studies.py
echo ""

# 鈹€鈹€ Step 9: SHACL + governance validation 鈹€鈹€
echo "[Step 9/12] Running SHACL + governance validation..."
$PYTHON reproduce_validation.py
echo ""

# 鈹€鈹€ Step 10: Ontology quality assessment 鈹€鈹€
echo "[Step 10/12] Running ontology quality assessment..."
$PYTHON reproduce_ontology_quality.py
echo ""

# 鈹€鈹€ Step 11: Pilot expert evaluation 鈹€鈹€
echo "[Step 11/12] Running pilot expert evaluation..."
$PYTHON reproduce_user_study.py
echo ""

# 鈹€鈹€ Step 12: AV-port generalizability check 鈹€鈹€
echo "[Step 12/12] Running AV-port generalizability check (搂8)..."
$PYTHON port_autonomous_vehicle/run_av_port.py
echo ""

echo "================================================"
echo "  All steps completed successfully."
echo "  JSON results: outputs/"
echo "  Benchmark data: data/"
echo "================================================"

