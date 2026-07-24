# SkyRwa Benchmark Generator

Reproducible synthetic benchmark for the paper:
**"SkyRwa: A Knowledge-Graph-Driven Flight-to-Asset Pipeline for Urban Air Mobility"**

---

## Why Synthetic Data

Due to current Chinese civil aviation regulations (CAAC), real operational UAM flight data
cannot be publicly shared. This benchmark is synthesized based on publicly available
regulatory frameworks (CAAC low-altitude airspace management regulations, 2024) and
operational parameters documented in published UAM trials.

All parameter ranges (altitude ceilings, NFZ buffer distances, BVLOS corridor lengths,
visibility minima) are drawn directly from CAAC AC-91-FS-2023-20 and the 2024 low-altitude
economy pilot-zone announcements. The anomaly taxonomy follows the fault classification
scheme in the CAAC UAM airworthiness guidelines (AC-21-AA-2024-51).

---

## Module Structure

```
benchmark_generator/
├── __init__.py
├── README.md               # this file
├── scenario_spec.py        # declarative scenario definitions
├── coverage_matrix.py      # coverage table generator
└── generate.py             # main pipeline driver
```

---

## Random Seed Policy

| Item | Value |
|------|-------|
| `RANDOM_SEED` | **42** |
| `BENCHMARK_VERSION` | **1.0.0** |
| Base timestamp | `2026-01-15T08:00:00Z` |

The seed governs **only** the random offset added to each flight's start time
(`random.randint(0, 500)` hours from base).  All other parameters are
deterministic functions of the scenario spec and the within-scenario flight
index `i`.  A different seed will shift timestamps but will not change
violation injection or parameter distributions.

---

## Scenario Generation Rules

Each scenario in `SCENARIO_SPECS` defines:

- **`count`** – number of flights to generate
- **`flight_id_template`** / **`uav_id_template`** – ID patterns with index `i`
  and UAV slot computed as `(i % uav_slot_modulus) + 1`
- **`parameter_distributions`** – per-field sampling rules (see below)
- **`injected_violations`** – list of `{violation, condition, rationale}` dicts
  where `condition` is a Python expression in `i`
- **`emergent_violations`** – pipeline-produced outcomes; recorded as
  `__emergent__<mechanism>` in `violation_tags` for post-run validation

---

## Parameter Distributions

| `dist` key | Semantics |
|------------|-----------|
| `constant` | Same value for all `i` |
| `linear`   | `base + i * step` |
| `cycle`    | `values[i % len(values)]` |
| `threshold` | `value` if `i >= threshold`, else `[]` / `below_value` |
| `threshold_bool` | `True` if `i < threshold`, else `False` |
| `threshold_value` | `below_value` if `i < threshold`, else `eval(above_expr)` |
| `threshold_ramp` | `max(0, i - threshold)` (used for NFZ incursion counts) |
| `threshold_bool_int` | `1` if `i >= threshold`, else `0` |

---

## Violation Classification

### Injected violations
Inserted deterministically by `generate.py` before the record enters the
pipeline.  Each entry in `injected_violations` carries a human-readable
`rationale` explaining why that violation models a real-world scenario.

### Emergent violations
Produced by the pipeline's own scoring logic (governance engine, anomaly
accumulator).  They are **not** pre-populated in the ingest record.
`emergent_violations` in the spec documents the triggering mechanism so
evaluators can validate that the pipeline produces the expected outcome.

The output file `benchmark_labels.json` records `violation_origin` for every
flight:

```json
{
  "flight_id": "FLT-NFZ-005",
  "violation_origin": {
    "nfz_proximity_warning": "injected"
  }
}
```

---

## Coverage Matrix

Run `python -m SkyRwa.benchmark_generator.coverage_matrix` to print:

```
+---------------------+--------------------+---------------------+...
| Scenario            | Structural Violation | Threshold Violation |...
+=====================+======================+=====================+...
| clean_route_survey  |          -           |          -          |...
| night_flight        |          Y           |          -          |...
...
```

The full matrix is also exported to `coverage_matrix.json` during generation.

| Scenario | Struct. | Threshold | Conditional | Emergent | Desensit. | Promotion | Std Gov | Rights | Mission fail | Quality fail | Promotion LC | Settlement | Rejection | Partial |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| clean_route_survey    | - | - | - | - | - | Y | - | - | - | - | Y | Y | - | - |
| night_flight          | Y | - | - | - | - | - | Y | - | - | - | Y | Y | - | - |
| weather_disturbance   | Y | - | - | - | Y | - | - | - | - | - | Y | Y | - | - |
| near_nfz              | - | Y | Y | - | - | - | Y | - | - | - | Y | - | Y | Y |
| anomaly_maintenance   | - | - | - | Y | - | - | Y | - | - | - | - | - | Y | - |
| emergency_logistics   | - | Y | Y | - | - | - | - | - | Y | - | - | - | Y | - |
| low_quality           | Y | - | Y | - | - | - | - | - | - | Y | - | - | Y | - |
| rights_conflict       | - | - | - | Y | - | - | - | Y | - | - | - | - | Y | - |
| beyond_vlos           | - | Y | Y | - | - | - | Y | - | - | - | Y | - | Y | Y |
| urban_corridor        | - | Y | - | Y | - | - | Y | - | Y | - | - | - | Y | - |

---

## Scenario Summaries

| # | Tag | Flights | Tradable | Governance path | Violation source |
|---|-----|--------:|---------:|-----------------|-----------------|
| 1 | `clean_route_survey`  | 12 | 12 | direct\_promotion       | none |
| 2 | `night_flight`        |  8 |  8 | standard\_governance    | none (structural anomaly) |
| 3 | `weather_disturbance` | 10 | 10 | desensitization\_gate   | none (structural anomaly) |
| 4 | `near_nfz`            |  8 |  3 | mixed pass/non-transfer | injected: `nfz_proximity_warning` (i≥3) |
| 5 | `anomaly_maintenance` | 10 |  0 | standard\_governance    | emergent: anomaly accumulation |
| 6 | `emergency_logistics` |  8 |  0 | mission\_failure        | injected: `altitude_exceedance` (i≥5) |
| 7 | `low_quality`         | 12 |  0 | quality\_failure        | injected: `data_gap` (all), `sensor_failure` (odd i) |
| 8 | `rights_conflict`     |  8 |  0 | aggregation edge case   | emergent: rights conflict at aggregation |
| 9 | `beyond_vlos`         | 15 | 12 | range/link edge case    | injected: `range_exceedance` (i≥12) |
|10 | `urban_corridor`      | 14 |  0 | urban\_density/NFZ      | injected: `altitude_exceedance` (i≥11); emergent: obstacle proximity (i≥8) |
| | **Total** | **105** | **45** | | |

---

## Output Files

After running `python -m SkyRwa.benchmark_generator.generate`:

| File | Contents |
|------|----------|
| `sample_data/benchmark_flights.json` | Materialised flight parameters for all 105 flights |
| `sample_data/benchmark_assets.json`  | Full `FlightAssetUnit` objects post-pipeline |
| `sample_data/benchmark_labels.json`  | Ground-truth labels + `violation_origin` tags |
| `sample_data/benchmark_summary.json` | Aggregate statistics |
| `sample_data/coverage_matrix.json`   | Machine-readable coverage matrix |
| `sample_graphs/benchmark_graph.ttl`  | RDF graph (Turtle) |
| `sample_graphs/benchmark_graph.jsonld` | RDF graph (JSON-LD) |

---

## Reproducing the Benchmark

```bash
cd SkyNetUamPlatform/modules
python -m SkyRwa.benchmark_generator.generate --output-dir ./benchmark_output
```

The generator will always produce byte-identical JSON outputs for a given
`BENCHMARK_VERSION` and `RANDOM_SEED`.  If you need a different output
location, pass `--output-dir`.  To regenerate with a different seed, edit
`RANDOM_SEED` in `scenario_spec.py` and update `BENCHMARK_VERSION`.
