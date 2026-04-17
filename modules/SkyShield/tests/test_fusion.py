"""Tests for multi-radar track-level fusion and handoff bookkeeping."""

from __future__ import annotations

import numpy as np

from skyshield.radar.fusion import TrackFusion


def _est(rid, pos, sigma, vel=(20.0, 0.0, 0.0), conf=0.9):
    cov = np.eye(3) * (sigma ** 2)
    return rid, np.array(pos, dtype=float), cov, np.array(vel, dtype=float), conf


def test_fusion_returns_none_on_empty_input():
    fusion = TrackFusion(method="covariance_weighted")
    assert fusion.fuse(target_id=1, per_radar_estimates=[], t_ms=0.0) is None


def test_covariance_weighted_fusion_reduces_variance():
    fusion = TrackFusion(method="covariance_weighted")
    estimates = [
        _est(0, (100.0, 0.0, 50.0), sigma=4.0),
        _est(1, (100.5, 0.5, 50.2), sigma=4.0),
        _est(2, ( 99.5, -0.5, 49.8), sigma=4.0),
    ]
    track = fusion.fuse(target_id=42, per_radar_estimates=estimates, t_ms=10.0)
    assert track is not None
    assert track.target_id == 42
    fused_var = float(np.trace(track.covariance))
    single_var = float(np.trace(estimates[0][2]))
    assert fused_var < single_var / 2.0, (
        f"covariance-weighted fusion did not shrink variance: "
        f"single={single_var:.3f} fused={fused_var:.3f}"
    )
    assert track.contributing_radars == (0, 1, 2)


def test_nearest_radar_picks_lowest_trace():
    fusion = TrackFusion(method="nearest_radar")
    estimates = [
        _est(0, (10.0, 10.0, 0.0), sigma=8.0),
        _est(1, (10.0, 10.0, 0.0), sigma=2.0),
        _est(2, (10.0, 10.0, 0.0), sigma=5.0),
    ]
    track = fusion.fuse(target_id=1, per_radar_estimates=estimates, t_ms=0.0)
    assert track.primary_radar == 1


def test_handoff_records_latency_when_primary_changes():
    fusion = TrackFusion(method="covariance_weighted")
    estimates_a = [
        _est(0, (0.0, 0.0, 0.0), sigma=1.0),
        _est(1, (0.0, 0.0, 0.0), sigma=4.0),
    ]
    fusion.fuse(target_id=7, per_radar_estimates=estimates_a, t_ms=100.0)
    assert fusion.last_handoff_ms == 0.0
    estimates_b = [
        _est(0, (0.0, 0.0, 0.0), sigma=4.0),
        _est(1, (0.0, 0.0, 0.0), sigma=1.0),
    ]
    fusion.fuse(target_id=7, per_radar_estimates=estimates_b, t_ms=125.0)
    assert fusion.last_handoff_ms > 0.0
    assert fusion.last_handoff_ms <= 35.0
