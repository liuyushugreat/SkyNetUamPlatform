# SkyCert one-click reproduction for ESORICS 2026 reviewers (PowerShell).
#
# What it does (runs in order, ~30 seconds on a single CPU core):
#   1. install dependencies,
#   2. run the unit test suite (9 tests),
#   3. run the main threat-scenario experiment,
#   4. run the ablation study,
#   5. render the three paper figures,
#   6. print a short summary of the key metrics.
#
# No GPU, API key, or network access is required.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "[SkyCert] Installing dependencies ..." -ForegroundColor Cyan
python -m pip install --quiet -r requirements.txt

Write-Host "[SkyCert] Running unit tests ..." -ForegroundColor Cyan
python -m pytest tests -q

Write-Host "[SkyCert] Running main threat-scenario experiment ..." -ForegroundColor Cyan
python -m scripts.run_experiment --config configs/default.yaml

Write-Host "[SkyCert] Running ablation study ..." -ForegroundColor Cyan
python -m scripts.run_ablation   --config configs/default.yaml

Write-Host "[SkyCert] Running baseline comparison ..." -ForegroundColor Cyan
python -m scripts.run_baselines  --config configs/default.yaml

Write-Host "[SkyCert] Running extension experiments (lambda sweep, attack-strength sweep, failure cases, MLP) ..." -ForegroundColor Cyan
python -m scripts.run_extensions --config configs/default.yaml

Write-Host "[SkyCert] Running 5-seed aggregation (Tables 1/2/3 mean+-std) ..." -ForegroundColor Cyan
python -m scripts.run_multi_seed --config configs/default.yaml

Write-Host "[SkyCert] Rendering paper figures ..." -ForegroundColor Cyan
python -m scripts.plot_results   --config configs/default.yaml

Write-Host "[SkyCert] Summary (see outputs/metrics.json for full data):" -ForegroundColor Cyan
$py = @'
import json, pathlib
p = pathlib.Path("outputs/metrics.json")
data = json.loads(p.read_text())
print(f"{'scenario':<22}{'coverage':>10}{'abstain':>10}{'crit_base':>12}{'crit_after':>12}{'M_max':>14}")
for r in data["runs"]:
    name = r["threat"]["name"]
    print(f"{name:<22}"
          f"{r['coverage']:>10.3f}"
          f"{r['abstain_rate']:>10.3f}"
          f"{r['critical_error_rate_base']:>12.3f}"
          f"{r['critical_error_rate_after_abstain']:>12.3f}"
          f"{r['martingale_max']:>14.2e}")
lat = [r["avg_decision_ms"] for r in data["runs"]]
print(f"\nmean end-to-end decision latency: {sum(lat)/len(lat):.3f} ms")
'@
python -c $py

Write-Host "[SkyCert] Done. Artifacts are in .\outputs\" -ForegroundColor Green
