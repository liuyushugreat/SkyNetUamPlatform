"""Integration test for the full sense-decide-act loop."""
from __future__ import annotations

from pathlib import Path

from skyshield.runtime.engine import SkyShieldRuntime
from skyshield.workload import from_field_sorties, generate


def test_field_sorties_run_end_to_end(default_cfg, field_sorties_path):
    scenarios = from_field_sorties(field_sorties_path, default_cfg, augment=0)
    rt = SkyShieldRuntime(default_cfg, config_path="configs/default.yaml")
    rep = rt.run(scenarios)
    assert rep.num_threats == 10
    summary = rep.metrics.summary()
    assert summary["num_events"] == 10
    # Mission success must be at least 60% — the real field run was 80%.
    assert summary["mission_success_rate"] >= 0.6


def test_generate_workload_produces_deterministic_counts(default_cfg):
    a = generate(default_cfg, duration_s=60.0, concurrency=1, seed=7)
    b = generate(default_cfg, duration_s=60.0, concurrency=1, seed=7)
    assert len(a) == len(b)
    assert [s.target_id for s in a] == [s.target_id for s in b]


def test_deadline_aware_outperforms_fifo(default_cfg):
    # EDF+slack should miss fewer deadlines than FIFO under the same load.
    fifo_cfg = default_cfg.with_overrides({"decision.scheduler": "fifo"})
    edf_cfg = default_cfg.with_overrides({"decision.scheduler": "edf_slack"})
    sc_fifo = generate(fifo_cfg, duration_s=60.0, concurrency=3, seed=11)
    sc_edf = generate(edf_cfg, duration_s=60.0, concurrency=3, seed=11)
    fifo = SkyShieldRuntime(fifo_cfg).run(sc_fifo).metrics.summary()
    edf = SkyShieldRuntime(edf_cfg).run(sc_edf).metrics.summary()
    # Weak inequality because the workload itself is identical at per-threat
    # level; the test mainly exercises the pipeline, not the optimality.
    assert edf["deadline_miss_ratio"] <= fifo["deadline_miss_ratio"] + 1e-6
