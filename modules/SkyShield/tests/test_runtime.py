"""End-to-end tests for the SkyShield discrete-event runtime."""

from __future__ import annotations

from pathlib import Path

import pytest

from skyshield.config import SkyShieldConfig
from skyshield.runtime.engine import RuntimeOptions, SkyShieldRuntime, SortieScenario


def _scen(**overrides) -> SortieScenario:
    base = dict(
        sortie_id=1,
        test_type="interception_test",
        target_takeoff_t="11:00",
        target_speed_kmh=140.0,
        target_height_m=120.0,
        interceptor_takeoff_t="11:05",
        is_real=True,
        target_maneuver_g=0.5,
        spawn_distance_m=5_000.0,
    )
    base.update(overrides)
    return SortieScenario(**base)


def _make_runtime(default_config, opts: RuntimeOptions, seed: int) -> SkyShieldRuntime:
    cfg = default_config
    # Apply the per-test seed without mutating the shared default_config fixture.
    from dataclasses import replace
    cfg = replace(cfg, seed=seed)
    return SkyShieldRuntime(cfg=cfg, opts=opts)


@pytest.fixture
def runtime(default_config) -> SkyShieldRuntime:
    return _make_runtime(default_config, RuntimeOptions(label="test"), seed=12345)


_VALID_OUTCOMES = {
    "hit_shot_down", "hit_partial", "hit_not_shot_down", "miss",
    "target_lost", "aborted", "abort_deadline_miss", "suppressed",
}


def test_runtime_executes_single_sortie(runtime):
    rec = runtime.run_sortie(_scen())
    assert rec.outcome in _VALID_OUTCOMES, f"unexpected outcome {rec.outcome!r}"
    assert rec.end_to_end_ms > 0.0
    assert rec.end_to_end_ms < 1500.0, "must respect the 1.5 s end-to-end deadline"


def test_runtime_replays_field_sortie_book(default_config, module_root: Path):
    import json
    book = json.loads((module_root / "data" / "field_sorties.json").read_text())
    runtime = _make_runtime(default_config, RuntimeOptions(label="real"), seed=2026)
    sorties = book["sorties"]
    success = 0
    for entry in sorties:
        scen = _scen(
            sortie_id=entry["sortie_id"],
            test_type=entry["test_type"],
            target_takeoff_t=entry["target_takeoff_t"],
            target_speed_kmh=float(entry.get("target_speed_kmh") or 130.0),
            target_height_m=float(entry.get("target_height_m") or 100.0),
            interceptor_takeoff_t=entry.get("interceptor_takeoff_t") or
                                  entry["target_takeoff_t"],
            is_real=True,
            forced_abort=(entry["outcome"] == "abort"),
            forced_lost_lock=(entry["outcome"] == "target_lost"),
            target_maneuver_g=0.6,
        )
        rec = runtime.run_sortie(scen)
        # A "valid" mission outcome includes both successful interceptions
        # (hit_*) and successful aborts.
        if rec.outcome.startswith("hit") or rec.outcome == "aborted":
            success += 1
    assert success >= 6, (
        f"replaying the 10 field sorties should succeed at least 6/10; got {success}"
    )
    assert runtime.metrics.deadline_misses == 0


def test_runtime_zero_false_launches_with_unauthorized_inputs(default_config):
    opts = RuntimeOptions(label="adversary", auth_grant_pct=0.0)
    runtime = _make_runtime(default_config, opts, seed=1)
    for sid in range(1, 41):
        runtime.run_sortie(_scen(sortie_id=sid))
    assert runtime.metrics.false_launch == 0
    assert runtime.metrics.successful_intercepts == 0
    assert runtime.metrics.suppressed >= 30


def test_disabling_scheduler_inflates_tail_latency(default_config):
    opts_a = RuntimeOptions(label="with_sched", enable_scheduler=True)
    opts_b = RuntimeOptions(label="no_sched", enable_scheduler=False)
    rt_a = _make_runtime(default_config, opts_a, seed=2)
    rt_b = _make_runtime(default_config, opts_b, seed=2)
    for sid in range(1, 61):
        rt_a.run_sortie(_scen(sortie_id=sid))
        rt_b.run_sortie(_scen(sortie_id=sid))
    e2e_a = sorted(rt_a.metrics.end_to_end_ms)
    e2e_b = sorted(rt_b.metrics.end_to_end_ms)
    p99_a = e2e_a[int(0.99 * (len(e2e_a) - 1))]
    p99_b = e2e_b[int(0.99 * (len(e2e_b) - 1))]
    assert p99_b > p99_a * 1.10, (
        f"disabling the deadline scheduler must inflate p99: "
        f"with={p99_a:.1f} ms without={p99_b:.1f} ms"
    )
