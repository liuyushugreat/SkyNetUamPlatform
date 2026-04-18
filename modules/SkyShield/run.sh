#!/usr/bin/env bash
# SkyShield one-click reproduction for RTSS 2026 reviewers.
#
# What it does (runs in order, ~5-10 minutes total on a single CPU core):
#   1. install dependencies,
#   2. run the unit test suite (21 tests),
#   3. E1 replay the 10 real sorties + 50 augmented,
#   4. E2 end-to-end timing experiment,
#   5. E3 replay-based stress regimes,
#   6. E4 multi-radar urban deployment sweep,
#   7. E5 ablation study,
#   8. E6 safety / failure analysis,
#   9. render all paper figures,
#   10. print a short summary of the key metrics.
#
# No GPU, API key, or network access is required.

set -euo pipefail

cd "$(dirname "$0")"

echo "[SkyShield] Installing dependencies ..."
python -m pip install --quiet -r requirements.txt

echo "[SkyShield] Running unit tests ..."
python -m pytest tests -q

echo "[SkyShield] E1 - Field replay ..."
python -m scripts.run_field_replay --config configs/default.yaml

echo "[SkyShield] E2 - End-to-end timing ..."
python -m scripts.run_timing --config configs/default.yaml

echo "[SkyShield] E3 - Replay-based stress ..."
python -m scripts.run_replay_stress --config configs/replay.yaml

echo "[SkyShield] E4 - Multi-radar deployment ..."
python -m scripts.run_multi_radar --config configs/multi_radar.yaml

echo "[SkyShield] E5 - Ablation study ..."
python -m scripts.run_ablation --config configs/ablation.yaml

echo "[SkyShield] E6 - Safety and failure analysis ..."
python -m scripts.run_safety --config configs/default.yaml

echo "[SkyShield] Rendering paper figures ..."
python -m scripts.plot_results --outputs outputs

echo "[SkyShield] Summary:"
python - <<'PY'
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
PY

echo "[SkyShield] Done. Artifacts are in ./outputs/"
