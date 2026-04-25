# SkyShield — Field-Anchored Real-Time Counter-UAV Interception Runtime

> **RTSS 2026 reviewers:** this directory is the complete, anonymous
> reproduction artifact for our submission
> *"SkyShield: A Field-Anchored Radar-Guided Real-Time Counter-UAV
> Interception System for Urban Low-Altitude Defense"*
> (`pressRequire/SkyShield/SkyShield_RTSS2026/`).
> Run `bash run.sh` on Linux/macOS or `./run.ps1` on Windows to
> regenerate every number, figure, and PDF used in the paper in
> under ten minutes on a single CPU core.  No GPU, no network, no
> API keys.

SkyShield is a deadline-aware, safety-guarded, radar-guided
real-time counter-UAV (C-UAV) interception runtime for a
$300\,\text{km}^2$ urban low-altitude district.  It is implemented
as a single-process deterministic **discrete-event simulator (DES)**
so that one seed reproduces every reported number bit-for-bit.

The runtime composes:

* a PLFM-style **multi-radar sensing plane** with $M$-of-$N$
  confirmation, covariance-weighted track-to-track fusion, and
  explicit handoff-latency accounting;
* a **deadline-aware decision plane** with FIFO / RM / EDF /
  EDF+slack-stealing policies, enforcing a $1.5$-second end-to-end
  deadline and a hard $200$-ms abort deadline;
* a **runtime safety guard** that gates every launch on
  authorization, geofence, friendly-airspace, and classification
  confidence preconditions;
* a **bounded fail-safe abort controller** with `return_safe`
  kinematics.

All six experiments in the paper (field replay, end-to-end timing,
replay-based stress, multi-radar urban deployment, ablation, safety)
are driven by the same runtime and aggregated by
`scripts/plot_results.py` into `outputs/*.json` and `outputs/*.pdf`.

## At a glance

| Contribution | Role | Measurable effect |
|---|---|---|
| **Three-plane architecture** | Explicit stage-budget table that composes $D_{e2e}=1500$ ms | End-to-end P99 **391 ms**; every stage P99 $\leq 2\times$ its P50 |
| **EDF + slack stealing** | Threat-prioritised queue under HIL authorization delay | **−22 %** P99 latency vs. FIFO under concurrency 4 |
| **Runtime safety guard** | Friendly-airspace / low-confidence / subthreshold checks before the launch gate | Correct response for five of six modeled scenarios over $6 \times 100$ binomial trials ($95 \%$ LCB $0.964$) |
| **Bounded fail-safe abort** | Engagement-progress-aware recall, refuses return-safe when $R_3$ would miss | **100 %** abort-within-deadline across the field-anchored replay workload |
| **Multi-radar urban deployment** | $300\,\text{km}^2$ district, $1$–$12$ radar sweep $\times$ $1$–$8$ target concurrency | Deadline miss $0.72 \to 0.00$ going $1 \to 6$ radars at concurrency $1$ |

End-to-end, on the $10$-field-sortie + $50$-augmented-sortie
workload at the default configuration:

| Metric                                 | Value |
|----------------------------------------|-------|
| Mission success rate                   | **0.68** |
| \etae P99 latency                      | **391 ms** (budget $1500$ ms) |
| Deadline miss ratio                    | **3.3 %** |
| Abort-within-deadline rate             | **1.00** |
| Return-safe rate after abort           | **1.00** |
| False-launch suppression rate          | **5.0 %** |
| Replay-stress worst-case P99 (auth-delay) | 588 ms (still under budget) |

## Repository layout

```
modules/SkyShield/
├── configs/
│   ├── default.yaml          # nominal 4-radar district + 1.5 s deadline
│   ├── multi_radar.yaml      # E4 deployment sweep parameters
│   ├── ablation.yaml         # E5 component toggles
│   └── replay.yaml           # E3 stress regimes
├── data/
│   ├── field_sorties.json    # 10 real interception sorties (verbatim)
│   └── augmented_seeds.json  # deterministic augmentation seeds
├── scripts/
│   ├── run_field_replay.py   # E1
│   ├── run_timing.py         # E2
│   ├── run_replay_stress.py  # E3
│   ├── run_multi_radar.py    # E4
│   ├── run_ablation.py       # E5
│   ├── run_safety.py         # E6
│   └── plot_results.py       # regenerates all paper figures
├── skyshield/
│   ├── radar/                # PLFM node + covariance-weighted fusion
│   ├── tracker/              # CV Kalman + M-of-N confirmation
│   ├── decision/             # threat scoring + deadline scheduler +
│   │                         # safety guard + bounded abort
│   ├── interceptor/          # kinematics model + launch gate
│   ├── runtime/              # DES engine and virtual clock
│   ├── telemetry/            # span tracer + RunMetrics
│   ├── workload.py           # field/augmented/synthetic generators
│   ├── config.py             # YAML loader + dataclasses
│   └── utils.py
├── tests/                    # 28 pytest cases
├── outputs/                  # generated JSON + PDF (gitignored)
├── run.sh / run.ps1          # one-click reproduction
├── pyproject.toml
└── requirements.txt
```

## One-click reproduction

```bash
cd modules/SkyShield
python -m pip install -r requirements.txt
bash run.sh          # Linux/macOS
./run.ps1            # Windows PowerShell
```

The script will, in order:

1. install dependencies;
2. run the `pytest` suite;
3. run experiments E1–E6 (`run_field_replay`, `run_timing`,
   `run_replay_stress`, `run_multi_radar`, `run_ablation`,
   `run_safety`) and write JSON outputs to `outputs/`;
4. regenerate every Matplotlib figure used in the paper
   (`fig_timing_cdf.pdf`, `fig_stress_tradeoff.pdf`,
   `fig_multi_radar_scaling.pdf`, `fig_ablation_bars.pdf`,
   `fig_safety_ci.pdf`) to `outputs/`;
5. print a summary table with the headline numbers.

Expect a total runtime of under ten minutes on a single CPU core.

## Running individual experiments

```bash
python scripts/run_field_replay.py                     # E1
python scripts/run_timing.py --duration 300            # E2
python scripts/run_replay_stress.py                    # E3
python scripts/run_multi_radar.py                      # E4
python scripts/run_ablation.py                         # E5
python scripts/run_safety.py                           # E6
python scripts/plot_results.py                         # figures
```

Every script accepts `--config configs/default.yaml` (or a sibling
config) and `--out outputs/<name>.json`.  Seeds flow through
`configs/default.yaml` (default `20260418`) and
`data/augmented_seeds.json`; a reviewer who changes a seed will
see a different sample path but the same contract-level numbers.

## Testing

```bash
pytest -q
```

Covers the Kalman tracker, M-of-N confirmer, covariance-weighted
fuser, radar node range/sigma model, deadline scheduler, safety
guard, bounded abort controller, interceptor kinematics, threat
scoring, and the full end-to-end runtime.  All tests are CPU-only
and deterministic.

## Mapping from artifact to paper

| Paper element                             | File                                            |
|-------------------------------------------|-------------------------------------------------|
| Fig. 1 – System architecture              | `pressRequire/SkyShield/.../fig_arch.pdf`       |
| Fig. 2 – Perception link                  | `pressRequire/SkyShield/.../fig_sensing.pdf`    |
| Fig. 3 – Real-time closed loop            | `pressRequire/SkyShield/.../fig_loop.pdf`       |
| Fig. 4 – E2 timing CDF                    | `outputs/fig_timing_cdf.pdf`                    |
| Fig. 5 – E3 stress trade-off              | `outputs/fig_stress_tradeoff.pdf`               |
| Fig. 6 – E4 multi-radar scaling           | `outputs/fig_multi_radar_scaling.pdf`           |
| Fig. 7 – Urban deployment                 | `pressRequire/SkyShield/.../fig_urban.pdf`      |
| Fig. 8 – E5 ablation bars                 | `outputs/fig_ablation_bars.pdf`                 |
| Fig. 9 – E6 safety CIs                    | `outputs/fig_safety_ci.pdf`                     |
| Tab. I – Stage budget                     | `skyshield/runtime/engine.py` (`_POLICY_MULT`)  |
| Tab. II – Field replay                    | `data/field_sorties.json` + `outputs/field_replay.json` |
| Tab. III – E2 stage latencies             | `outputs/timing.json`                           |
| Tab. IV – E3 stress regimes               | `outputs/replay_stress.json`                    |
| Tab. V – E4 deadline-miss sweep           | `outputs/multi_radar.json`                      |
| Tab. VI – E5 ablation                     | `outputs/ablation.json`                         |
| Tab. VII – E6 safety CIs                  | `outputs/safety.json`                           |
| Compiled paper PDF                        | `pressRequire/SkyShield/SkyShield_RTSS2026/skyshield_rtss2026.pdf` |

## Extending SkyShield

* **New scheduler policy.** Add a branch to
  `skyshield/decision/deadline.py::DeadlineScheduler.pick_next`
  and a corresponding entry in
  `skyshield/runtime/engine.py::_POLICY_MULT`.
* **New radar physics.** Replace
  `skyshield/radar/node.py::RadarNode.observe`.  The fuser,
  tracker and confirmer are agnostic to the underlying physics.
* **New safety scenario.** Add a scenario factory to
  `scripts/run_safety.py` (see
  `_friendly_airspace`, `_low_confidence`, ...) and re-run
  E6; the binomial CI machinery will pick it up automatically.

## Reproducibility contract

Every number in the paper (abstract, tables, figures, footnotes) is
a direct output of the scripts above running against
`configs/default.yaml` and `data/augmented_seeds.json`.
`SkyShieldRuntime` emits a `RunMetrics` dataclass that serialises
to JSON verbatim — reviewers can diff two runs field-by-field.

## License

This project is licensed under the Apache License 2.0, matching
the rest of the SkyNetUAM platform.
