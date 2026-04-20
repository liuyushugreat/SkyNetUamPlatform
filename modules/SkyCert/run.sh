#!/usr/bin/env bash
# SkyCert one-click reproduction for ESORICS 2026 reviewers.
#
# What it does (runs in order, ~30 seconds total on a single CPU core):
#   1. install dependencies,
#   2. run the unit test suite (9 tests),
#   3. run the main threat-scenario experiment     -> outputs/metrics.json,
#   4. run the ablation study                      -> outputs/ablation.json,
#   5. render the three paper figures              -> outputs/figs/,
#   6. print a short summary of the key metrics.
#
# No GPU, API key, or network access is required.

set -euo pipefail

cd "$(dirname "$0")"

echo "[SkyCert] Installing dependencies ..."
python -m pip install --quiet -r requirements.txt

echo "[SkyCert] Running unit tests ..."
python -m pytest tests -q

echo "[SkyCert] Running main threat-scenario experiment ..."
python -m scripts.run_experiment --config configs/default.yaml

echo "[SkyCert] Running ablation study ..."
python -m scripts.run_ablation   --config configs/default.yaml

echo "[SkyCert] Running baseline comparison ..."
python -m scripts.run_baselines  --config configs/default.yaml

echo "[SkyCert] Rendering paper figures ..."
python -m scripts.plot_results   --config configs/default.yaml

echo "[SkyCert] Summary (see outputs/metrics.json for full data):"
python - <<'PY'
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
PY

echo "[SkyCert] Done. Artifacts are in ./outputs/"
