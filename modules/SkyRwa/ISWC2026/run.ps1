# ============================================================
# SkyRwa 鈥?ISWC 2026 One-Click Reproduction Script (Windows)
# Paper: "Modeling Governable Flight-to-Asset Lifecycles
#         with Knowledge Graphs, SHACL, and Provenance"
# ============================================================

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Detect Python
$Python = $null
foreach ($cmd in @("python", "python3")) {
    try {
        $ver = & $cmd --version 2>&1
        $Python = $cmd
        Write-Host "[INFO] Using Python: $ver"
        break
    } catch {}
}
if (-not $Python) {
    Write-Host "[ERROR] Python not found. Please install Python 3.10+."
    exit 1
}

# Install dependencies
if (Test-Path "requirements.txt") {
    Write-Host "[Step 0] Installing dependencies..."
    & $Python -m pip install -r requirements.txt --quiet
}

Write-Host ""
Write-Host "================================================"
Write-Host "  SkyRwa ISWC 2026 Artifact Reproduction"
Write-Host "================================================"
Write-Host ""

# Step 1: Generate benchmark dataset (Table 5)
Write-Host "[Step 1/12] Generating benchmark dataset (Table 5: 105 flights, 10 scenarios)..."
& $Python reproduce_table5.py
Write-Host ""

# Step 2: Baseline comparison (Table 6)
Write-Host "[Step 2/12] Running baseline comparison: JSON-scan vs SPARQL (Table 6)..."
& $Python reproduce_table6.py
Write-Host ""

# Step 3: Semantic baseline (Lifecycle KG vs Flat KG)
Write-Host "[Step 3/12] Running semantic baseline: Lifecycle KG vs Flat KG..."
& $Python reproduce_semantic_baseline.py
Write-Host ""

# Step 4: Governance ablation (Table 7)
Write-Host "[Step 4/12] Running governance ablation: Python vs SHACL vs Combined (Table 7)..."
& $Python reproduce_table7.py
Write-Host ""

# Step 5: Scalability (Table 8)
Write-Host "[Step 5/12] Running scalability experiment: 5-1000 flights (Table 8)..."
& $Python reproduce_table8.py
Write-Host ""

# Step 6: SPARQL competency questions (Table 9)
Write-Host "[Step 6/12] Running SPARQL competency questions (Table 9)..."
& $Python reproduce_table9.py
Write-Host ""

# Step 7: Robustness (multi-run, scale, thresholds)
Write-Host "[Step 7/12] Running robustness experiments..."
& $Python reproduce_robustness.py
Write-Host ""

# Step 8: Case studies
Write-Host "[Step 8/12] Running case studies..."
& $Python reproduce_case_studies.py
Write-Host ""

# Step 9: SHACL + governance validation
Write-Host "[Step 9/12] Running SHACL + governance validation..."
& $Python reproduce_validation.py
Write-Host ""

# Step 10: Ontology quality assessment
Write-Host "[Step 10/12] Running ontology quality assessment..."
& $Python reproduce_ontology_quality.py
Write-Host ""

# Step 11: Pilot expert evaluation
Write-Host "[Step 11/12] Running pilot expert evaluation..."
& $Python reproduce_user_study.py
Write-Host ""

# Step 12: AV-port generalizability check (Sect. 8)
Write-Host "[Step 12/12] Running AV-port generalizability check (Sect. 8)..."
& $Python port_autonomous_vehicle/run_av_port.py
Write-Host ""

Write-Host "================================================"
Write-Host "  All steps completed successfully."
Write-Host "  JSON results: outputs/"
Write-Host "  Benchmark data: data/"
Write-Host "================================================"

