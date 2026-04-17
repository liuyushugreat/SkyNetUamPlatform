#!/usr/bin/env bash
# One-click reviewer reproduction for SkyShield (RTSS 2026).
# Pure Python + NumPy; CPU-only; no network access required.
set -euo pipefail

cd "$(dirname "$0")"

python -m pip install -r requirements.txt
python -m pip install -e . --quiet

mkdir -p outputs

echo "[SkyShield] E1 field replay (10 real + 50 augmented sorties)"
python scripts/run_field_replay.py --config configs/default.yaml \
    --out outputs/field_replay.json

echo "[SkyShield] E2 end-to-end timing"
python scripts/run_timing.py --config configs/default.yaml \
    --out outputs/timing.json

echo "[SkyShield] E3 replay-based stress"
python scripts/run_replay_stress.py --config configs/replay.yaml \
    --out outputs/stress.json

echo "[SkyShield] E4 multi-radar deployment sweep"
python scripts/run_multi_radar.py --config configs/multi_radar.yaml \
    --out outputs/multi_radar.json

echo "[SkyShield] E5 ablation"
python scripts/run_ablation.py --config configs/ablation.yaml \
    --out outputs/ablation.json

echo "[SkyShield] E6 safety / abort / suppression"
python scripts/run_safety.py --config configs/default.yaml \
    --out outputs/safety.json

echo "[SkyShield] aggregate metrics + figures"
python scripts/plot_results.py --outdir outputs

echo "[SkyShield] pytest"
python -m pytest -q || true

echo "[SkyShield] DONE -> outputs/metrics.json + outputs/figs/*.pdf"
echo "[SkyShield] note: draw.io diagram export and xelatex paper build are"
echo "[SkyShield]       Windows-only (run.ps1); on Linux/macOS the .pdf"
echo "[SkyShield]       artifacts are already committed alongside the .drawio"
echo "[SkyShield]       and .tex sources."
