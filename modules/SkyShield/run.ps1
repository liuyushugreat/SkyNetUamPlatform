# SkyShield one-click reproduction for RTSS 2026 reviewers (PowerShell).

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "[SkyShield] Installing dependencies ..." -ForegroundColor Cyan
python -m pip install --quiet -r requirements.txt

Write-Host "[SkyShield] Running unit tests ..." -ForegroundColor Cyan
python -m pytest tests -q

Write-Host "[SkyShield] E1 - Field replay ..." -ForegroundColor Cyan
python -m scripts.run_field_replay --config configs/default.yaml

Write-Host "[SkyShield] E2 - End-to-end timing ..." -ForegroundColor Cyan
python -m scripts.run_timing --config configs/default.yaml

Write-Host "[SkyShield] E3 - Replay-based stress ..." -ForegroundColor Cyan
python -m scripts.run_replay_stress --config configs/replay.yaml

Write-Host "[SkyShield] E4 - Multi-radar deployment ..." -ForegroundColor Cyan
python -m scripts.run_multi_radar --config configs/multi_radar.yaml

Write-Host "[SkyShield] E5 - Ablation study ..." -ForegroundColor Cyan
python -m scripts.run_ablation --config configs/ablation.yaml

Write-Host "[SkyShield] E6 - Safety and failure analysis ..." -ForegroundColor Cyan
python -m scripts.run_safety --config configs/default.yaml

Write-Host "[SkyShield] Rendering paper figures ..." -ForegroundColor Cyan
python -m scripts.plot_results --outputs outputs

Write-Host "[SkyShield] Summary:" -ForegroundColor Cyan
$py = @'
import json, pathlib
p = pathlib.Path("outputs/metrics.json")
if not p.exists():
    print("outputs/metrics.json not found"); raise SystemExit(0)
data = json.loads(p.read_text())
s = data["summary"]
print(f"  mission_success      = {s['mission_success_rate']:.3f}")
print(f"  valid_intercept      = {s['valid_intercept_rate']:.3f}")
print(f"  shot_down            = {s['shot_down_rate']:.3f}")
print(f"  end_to_end_p50_ms    = {s['latency_ms']['p50']:.1f}")
print(f"  end_to_end_p95_ms    = {s['latency_ms']['p95']:.1f}")
print(f"  end_to_end_p99_ms    = {s['latency_ms']['p99']:.1f}")
print(f"  deadline_miss_ratio  = {s['deadline_miss_ratio']:.4f}")
print(f"  abort_success        = {s['abort_success_rate']:.3f}")
print(f"  false_launch_suppr   = {s['false_launch_suppression_rate']:.4f}")
'@
python -c $py

Write-Host "[SkyShield] Done. Artifacts are in .\outputs\" -ForegroundColor Green
