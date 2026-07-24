# ============================================================
# SkyRwa -- One-Command Artifact Reproduction (JWS submission)
# Paper: "Modeling Governable Flight-to-Asset Lifecycles
#         with Knowledge Graphs, SHACL, and Provenance"
#
# Expected total runtime: ~10 minutes on a laptop-class CPU
# (step 5, the scoring-context cost study, takes ~6 minutes).
# ============================================================

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "[INFO] Using Python: $(python --version)"

if (Test-Path requirements.txt) {
    Write-Host "[Step 0] Installing dependencies..."
    python -m pip install -r requirements.txt --quiet
}

Write-Host ""
Write-Host "================================================"
Write-Host "  SkyRwa Artifact Reproduction (JWS submission)"
Write-Host "================================================"
Write-Host ""

$steps = @(
    @("reproduce_table5.py",           "Benchmark dataset (Table 5: 105 flights, 10 scenarios)"),
    @("reproduce_table6.py",           "JSON-scan vs SPARQL baseline (Table 6)"),
    @("reproduce_semantic_baseline.py","Lifecycle KG vs flat KG (Table 7)"),
    @("reproduce_table7.py",           "Governance ablation (Table 8)"),
    @("reproduce_scoring_context.py",  "Scoring-context materialization cost (Table 9, ~6 min)"),
    @("reproduce_table8.py",           "Runtime overhead 5-1000 flights (Table 10)"),
    @("reproduce_shacl_engines.py",    "Dual-engine SHACL comparison (Table 10 + Fig. 2)"),
    @("reproduce_robustness.py",       "Robustness: seeds, scale, thresholds (Sect. 7.6)"),
    @("reproduce_competency.py",       "Competency questions CQ1-CQ12 (Table 11)"),
    @("reproduce_walkthrough.py",      "End-to-end walkthrough (Sect. 7.8)"),
    @("reproduce_validation.py",       "SHACL + governance validation (Sect. 5)"),
    @("reproduce_ontology_quality.py", "Ontology quality + CQ mapping (Sect. 4.5 + Appendix A)"),
    @("port_autonomous_vehicle/run_av_port.py", "AV-port generalizability check (Sect. 8.1)")
)

$i = 0
foreach ($step in $steps) {
    $i++
    Write-Host "[Step $i/$($steps.Count)] $($step[1])..."
    python $step[0]
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $($step[0])" }
    Write-Host ""
}

Write-Host "================================================"
Write-Host "  All steps completed successfully."
Write-Host "  JSON results: outputs/"
Write-Host "  Benchmark data: data/"
Write-Host "================================================"
