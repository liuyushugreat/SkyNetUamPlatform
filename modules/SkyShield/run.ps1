#!/usr/bin/env pwsh
# One-click reviewer reproduction for SkyShield (RTSS 2026) on Windows.
$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

python -m pip install -r requirements.txt
python -m pip install -e . --quiet

New-Item -ItemType Directory -Force -Path outputs | Out-Null

Write-Host "[SkyShield] E1 field replay (10 real + 50 augmented sorties)"
python scripts/run_field_replay.py --config configs/default.yaml `
    --out outputs/field_replay.json

Write-Host "[SkyShield] E2 end-to-end timing"
python scripts/run_timing.py --config configs/default.yaml `
    --out outputs/timing.json

Write-Host "[SkyShield] E3 replay-based stress"
python scripts/run_replay_stress.py --config configs/replay.yaml `
    --out outputs/stress.json

Write-Host "[SkyShield] E4 multi-radar deployment sweep"
python scripts/run_multi_radar.py --config configs/multi_radar.yaml `
    --out outputs/multi_radar.json

Write-Host "[SkyShield] E5 ablation"
python scripts/run_ablation.py --config configs/ablation.yaml `
    --out outputs/ablation.json

Write-Host "[SkyShield] E6 safety / abort / suppression"
python scripts/run_safety.py --config configs/default.yaml `
    --out outputs/safety.json

Write-Host "[SkyShield] aggregate metrics + figures"
python scripts/plot_results.py --outdir outputs

Write-Host "[SkyShield] pytest"
python -m pytest -q

$drawio = "C:\Program Files\draw.io\draw.io.exe"
if (Test-Path $drawio) {
    Write-Host "[SkyShield] exporting draw.io diagrams to PDF"
    $prevPref = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    foreach ($name in @("arch","sensing","loop","test","urban")) {
        & $drawio -x -f pdf -o "diagrams\$name.pdf" "diagrams\$name.drawio" 2>$null | Out-Null
    }
    $ErrorActionPreference = $prevPref
} else {
    Write-Host "[SkyShield] draw.io desktop not found; skipping diagram export"
}

$xelatex = "C:\texlive\2025\bin\windows\xelatex.exe"
$bibtex  = "C:\texlive\2025\bin\windows\bibtex.exe"
if ((Test-Path $xelatex) -and (Test-Path $bibtex)) {
    Write-Host "[SkyShield] compiling paper PDF"
    Push-Location paper
    & $xelatex -interaction=nonstopmode SkyShield_RTSS2026.tex | Out-Null
    & $bibtex SkyShield_RTSS2026 | Out-Null
    & $xelatex -interaction=nonstopmode SkyShield_RTSS2026.tex | Out-Null
    & $xelatex -interaction=nonstopmode SkyShield_RTSS2026.tex | Out-Null
    Pop-Location
} else {
    Write-Host "[SkyShield] xelatex not found at C:\texlive\2025; skipping paper build"
}

Write-Host "[SkyShield] DONE -> outputs/metrics.json + outputs/figs/*.pdf + paper/SkyShield_RTSS2026.pdf"
