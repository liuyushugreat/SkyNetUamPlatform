# ============================================================
# SkyRwa — ISWC 2026 One-Click Reproduction Script (Windows)
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
Write-Host "[Step 1/8] Generating benchmark dataset (Table 5: 105 flights, 10 scenarios)..."
& $Python reproduce_table5.py
Write-Host ""

# Step 2: Baseline comparison (Table 6)
Write-Host "[Step 2/8] Running baseline comparison: JSON-scan vs SPARQL (Table 6)..."
& $Python reproduce_table6.py
Write-Host ""

# Step 3: Governance ablation (Table 7)
Write-Host "[Step 3/8] Running governance ablation: Python vs SHACL vs Combined (Table 7)..."
& $Python reproduce_table7.py
Write-Host ""

# Step 4: Scalability (Table 8)
Write-Host "[Step 4/8] Running scalability experiment: 5-1000 flights (Table 8)..."
& $Python reproduce_table8.py
Write-Host ""

# Step 5: SPARQL competency questions (Table 9)
Write-Host "[Step 5/8] Running SPARQL competency questions (Table 9)..."
& $Python reproduce_table9.py
Write-Host ""

# Step 6: Robustness (multi-run, scale, thresholds)
Write-Host "[Step 6/8] Running robustness experiments..."
& $Python reproduce_robustness.py
Write-Host ""

# Step 7: Case studies (Section 7.7)
Write-Host "[Step 7/8] Running case studies (Section 7.7)..."
& $Python reproduce_case_studies.py
Write-Host ""

# Step 8: SHACL + governance validation
Write-Host "[Step 8/8] Running SHACL + governance validation..."
& $Python reproduce_validation.py
Write-Host ""

Write-Host "================================================"
Write-Host "  All steps completed successfully."
Write-Host "  JSON results: outputs/"
Write-Host "  Benchmark data: data/"
Write-Host "================================================"
