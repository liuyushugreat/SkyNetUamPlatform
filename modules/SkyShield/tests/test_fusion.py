from __future__ import annotations

import numpy as np

from skyshield.config import RadarConfig
from skyshield.radar.fusion import MultiRadarFuser
from skyshield.radar.node import RadarPacket


def _cfg(fusion_enabled: bool = True) -> RadarConfig:
    return RadarConfig(
        count=3, coverage_km=6.0, range_km_max=9.0, dwell_ms=18.0,
        revisit_ms=35.0, packet_mean_ms=5.0, packet_jitter_ms=2.0,
        dropout_rate=0.0, placement=[(0.0, 0.0)] * 3,
        fusion_enabled=fusion_enabled,
    )


def _pkt(node_id: int, t: float, tid: int, offset: float = 0.0) -> RadarPacket:
    return RadarPacket(
        node_id=node_id, emit_time_ms=t, arrive_time_ms=t + 1.0,
        target_id=tid, position_m=(100.0 + offset, 50.0, 120.0),
        velocity_mps=(30.0, 0.0, 0.0), snr_db=20.0, valid=True,
        meas_sigma_m=5.0,
    )


def test_fusion_combines_two_radars():
    f = MultiRadarFuser(_cfg(fusion_enabled=True))
    f.ingest(_pkt(0, 10.0, 1, offset=0.0))
    tr = f.ingest(_pkt(1, 12.0, 1, offset=2.0))
    assert tr is not None
    assert 0 in tr.contributing_nodes and 1 in tr.contributing_nodes
    assert tr.handoff_latency_ms > 0          # handoff measured


def test_fusion_disabled_overwrites_track():
    f = MultiRadarFuser(_cfg(fusion_enabled=False))
    t0 = f.ingest(_pkt(0, 10.0, 1, offset=0.0))
    t1 = f.ingest(_pkt(1, 12.0, 1, offset=5.0))
    assert t1 is not None
    assert t1.contributing_nodes == [1]       # fusion off -> overwrites
    assert t1 is not t0                        # fresh object per packet
