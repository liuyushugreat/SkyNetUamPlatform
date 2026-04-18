from __future__ import annotations

import numpy as np

from skyshield.config import RadarConfig
from skyshield.radar.node import RadarNode


def _cfg():
    return RadarConfig(
        count=1, coverage_km=6.0, range_km_max=9.0, dwell_ms=18.0,
        revisit_ms=35.0, packet_mean_ms=5.0, packet_jitter_ms=1.0,
        dropout_rate=0.0, placement=[(0.0, 0.0)],
    )


def test_out_of_range_returns_none():
    node = RadarNode(0, (0.0, 0.0), _cfg())
    rng = np.random.default_rng(0)
    pkt = node.observe(
        now_ms=0.0, target_id=1,
        target_pos_m=(20_000.0, 20_000.0, 120.0),  # ~28 km away
        target_vel_mps=(30.0, 0.0, 0.0), rng=rng,
    )
    assert pkt is None


def test_respects_revisit():
    node = RadarNode(0, (0.0, 0.0), _cfg())
    rng = np.random.default_rng(0)
    p1 = node.observe(0.0, 1, (3000.0, 0.0, 120.0), (30.0, 0.0, 0.0), rng)
    p2 = node.observe(5.0, 1, (3100.0, 0.0, 120.0), (30.0, 0.0, 0.0), rng)
    assert p1 is not None and p2 is None     # revisit gates the second


def test_measurement_sigma_grows_with_range():
    node = RadarNode(0, (0.0, 0.0), _cfg())
    close = node.measurement_sigma(1.0)
    far = node.measurement_sigma(8.0)
    assert far > close
