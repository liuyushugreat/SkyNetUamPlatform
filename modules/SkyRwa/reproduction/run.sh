#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# SkyRwa -- One-Command Artifact Reproduction (JWS submission)
# Paper: "Modeling Governable Flight-to-Asset Lifecycles
#         with Knowledge Graphs, SHACL, and Provenance"
#
# Expected total runtime: ~10 minutes on a laptop-class CPU
# (step 5, the scoring-context cost study, takes ~6 minutes).
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# -- Detect Python --
if command -v python3 &>/dev/null; then
  PYTHON=python3
elif command -v python &>/dev/null; then
  PYTHON=python
else
  echo "[ERROR] Python not found. Please install Python 3.10+."
  exit 1
fi
echo "[INFO] Using Python: $($PYTHON --version)"

# -- Install dependencies --
if [ -f requirements.txt ]; then
  echo "[Step 0] Installing dependencies..."
  $PYTHON -m pip install -r requirements.txt --quiet
fi

echo ""
echo "================================================"
echo "  SkyRwa Artifact Reproduction (JWS submission)"
echo "================================================"
echo ""

echo "[Step  1/13] Benchmark dataset (Table 5: 105 flights, 10 scenarios)..."
$PYTHON reproduce_table5.py
echo ""

echo "[Step  2/13] JSON-scan vs SPARQL baseline (Table 6)..."
$PYTHON reproduce_table6.py
echo ""

echo "[Step  3/13] Lifecycle KG vs flat KG (Table 7)..."
$PYTHON reproduce_semantic_baseline.py
echo ""

echo "[Step  4/13] Validation-layer coverage: Python vs SHACL vs Combined (Tables 8-9)..."
$PYTHON reproduce_table7.py
echo ""

echo "[Step  5/13] Scoring-context materialization cost (Table 10, ~6 min)..."
$PYTHON reproduce_scoring_context.py
echo ""

echo "[Step  6/13] Runtime overhead 5-1000 flights (Table 11)..."
$PYTHON reproduce_table8.py
echo ""

echo "[Step  7/13] Dual-engine SHACL comparison, pySHACL vs rudof (Table 11 + Fig. 2)..."
$PYTHON reproduce_shacl_engines.py
echo ""

echo "[Step  8/13] Robustness: seeds, scale, threshold sweep (Sect. 7.6)..."
$PYTHON reproduce_robustness.py
echo ""

echo "[Step  9/13] Competency-question verification, CQ1-CQ12 (Table 12)..."
$PYTHON reproduce_competency.py
echo ""

echo "[Step 10/13] End-to-end walkthrough of one blocked flight (Sect. 7.8)..."
$PYTHON reproduce_walkthrough.py
echo ""

echo "[Step 11/13] SHACL + governance validation coverage (Sect. 5)..."
$PYTHON reproduce_validation.py
echo ""

echo "[Step 12/13] Ontology quality: pitfalls, consistency, CQ mapping (Sect. 4.5 + Appendix A)..."
$PYTHON reproduce_ontology_quality.py
echo ""

echo "[Step 13/13] AV-port generalizability check (Sect. 8.1)..."
$PYTHON port_autonomous_vehicle/run_av_port.py
echo ""

echo "================================================"
echo "  All steps completed successfully."
echo "  JSON results: outputs/"
echo "  Benchmark data: data/"
echo "================================================"
