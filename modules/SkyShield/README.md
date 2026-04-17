# SkyShield

> **RTSS 2026 reviewers:** this directory is the complete, anonymous
> reproduction artifact for our submission *"SkyShield: A Deadline-Aware,
> Safety-Guarded Counter-UAV Interception Runtime Validated on a 10-Sortie
> Field Trial and a $300\,\text{km}^2$ Urban Replay Benchmark"*. Run
> `bash run.sh` on Linux/macOS or `.\run.ps1` on Windows to regenerate every
> paper number, figure, and PDF in under a minute on a laptop.

SkyShield is a deadline-aware runtime for radar-guided kinetic
counter-UAV (C-UAV) interception over a $300\,\text{km}^2$ urban area.
It is built as a single-process, deterministic discrete-event simulator so
that one seed reproduces every reported number byte-for-byte.

The runtime composes:

* a PLFM-style multi-radar sensing link with $M$-of-$N$ confirmation,
  covariance-weighted track-to-track fusion, and explicit handoff-latency
  accounting;
* a Rate-Monotonic + EDF + slack-stealing deadline scheduler that enforces
  a 1.5 s end-to-end deadline and a 200 ms hard abort deadline;
* a runtime safety guard that gates every launch on authorization,
  geofence, friendly-airspace, and classification-confidence
  preconditions;
* an abort controller with `return_safe` kinematics.

All six experiment axes in the paper (field replay, end-to-end timing,
replay-based stress, multi-radar urban scaling, ablation, safety) are
driven by the same runtime and aggregated by a single
`plot_results.py` script into `outputs/metrics.json`.

## Repository layout

```
modules/SkyShield/
├── configs/                # YAML configs: default / multi_radar / ablation / replay
├── data/                   # field_sorties.json (10 real sorties) + augmented_seeds.json (50)
├── diagrams/               # 5 draw.io architecture diagrams + PDF exports
├── outputs/                # generated JSON metrics, figures, and aggregate metrics.json
├── paper/                  # IEEE-conference anonymous source + compiled PDF
├── scripts/                # run_field_replay / run_timing / run_replay_stress /
│                           # run_multi_radar / run_ablation / run_safety / plot_results
├── skyshield/              # the Python package
│   ├── radar/              #   PLFM-style sensing + covariance-weighted fusion
│   ├── tracker/            #   Kalman CV + IMM + M-of-N confirmation
│   ├── decision/           #   threat scoring + RM/EDF/slack + safety guard + abort
│   ├── interceptor/        #   kinematic Ph model + launch controller
│   ├── runtime/            #   the discrete-event engine
│   └── telemetry/          #   Tracer + RunMetrics + SortieRecord
├── tests/                  # pytest suite: kalman, fusion, scheduler, abort, end-to-end
├── run.sh / run.ps1        # one-click reproduction
├── pyproject.toml
└── requirements.txt
```

## One-click reproduction

```bash
cd modules/SkyShield
python -m pip install -r requirements.txt
bash run.sh               # or:  .\run.ps1     (Windows PowerShell)
```

The script will, in order:

1. run each of the six experiment scripts
   (`run_field_replay`, `run_timing`, `run_replay_stress`,
   `run_multi_radar`, `run_ablation`, `run_safety`);
2. aggregate the per-axis outputs into `outputs/metrics.json` and plot
   `fig_cdf.pdf`, `fig_tail.pdf`, `fig_scaling.pdf`, `fig_failure.pdf`;
3. run the 24-test `pytest` suite;
4. export the five `.drawio` diagrams to PDF via
   `C:\Program Files\draw.io\draw.io.exe`;
5. compile the paper with `xelatex + bibtex + xelatex + xelatex`.

Expect a runtime of ≲ 60 s on a single CPU core.

## Headline numbers (from `outputs/metrics.json`)

| metric                                  | value           |
| --------------------------------------- | --------------- |
| real-field (10-sortie) success rate     | **80 %**        |
| augmented (50-sortie) success rate      | 70 %            |
| $N=900$ end-to-end mean latency         | **405 ms**      |
| $N=900$ end-to-end $p_{99}$             | **491 ms**      |
| end-to-end deadline misses ($N=900$)    | **0**           |
| false-launch suppression (adversary)    | **100 %**       |
| operator-abort success within 200 ms    | **100 %** (78/78)|
| $p_{99}$ drop, 1→12 radars, $T=8$       | $-8$ % (583→536 ms) |
| handoff drop, 1→12 radars, $T=8$        | $-68$ % (28→8 ms)   |
| $p_{99}$ inflation with no scheduler    | $+53$ % (497→761 ms)|

## Running individual experiments

```bash
python scripts/run_field_replay.py          # E1 field replay
python scripts/run_timing.py --repeats 15   # E2 end-to-end timing (900 sorties)
python scripts/run_replay_stress.py         # E3 stress regimes
python scripts/run_multi_radar.py           # E4 urban scaling
python scripts/run_ablation.py              # E5 ablation
python scripts/run_safety.py                # E6 safety & failure
python scripts/plot_results.py              # aggregate + figures
```

All scripts accept a `--seed` flag (default: 20260418, from
`configs/default.yaml`).

## Testing

```bash
pytest                    # 24 tests: kalman, fusion, scheduler, abort, runtime
```

## Mapping from artifact to paper

| paper element             | file                                     |
| ------------------------- | ---------------------------------------- |
| Fig. 1 (architecture)     | `diagrams/arch.drawio` / `arch.pdf`      |
| Fig. 2 (sensing)          | `diagrams/sensing.drawio` / `sensing.pdf`|
| Fig. 3 (real-time loop)   | `diagrams/loop.drawio` / `loop.pdf`      |
| Fig. 4 (urban deployment) | `diagrams/urban.drawio` / `urban.pdf`    |
| Fig. 5 (CDF)              | `outputs/figs/fig_cdf.pdf`               |
| Fig. 6 (tail under stress)| `outputs/figs/fig_tail.pdf`              |
| Fig. 7 (scaling)          | `outputs/figs/fig_scaling.pdf`           |
| Fig. 8 (failure)          | `outputs/figs/fig_failure.pdf`           |
| Fig. 9 (experiment matrix)| `diagrams/test.drawio` / `test.pdf`      |
| Tab. I (field replay)     | `data/field_sorties.json`                |
| Tab. II (timing)          | `outputs/timing.json`                    |
| Tab. III (stress)         | `outputs/stress.json`                    |
| Tab. IV (scaling)         | `outputs/multi_radar.json`               |
| Tab. V (ablation)         | `outputs/ablation.json`                  |
| Tab. VI (safety)          | `outputs/safety.json`                    |
| Compiled paper            | `paper/SkyShield_RTSS2026.pdf`           |

## License

Apache 2.0, matching the rest of the SkyNetUAM platform.
