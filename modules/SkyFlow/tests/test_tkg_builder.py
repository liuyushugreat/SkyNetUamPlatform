"""Tests for Temporal Knowledge Graph builder."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import pytest

from skyflow.data.tkg_builder import TKGBuilder, AirspaceState, TKGSnapshot


def _make_state(n_uav=10, n_sec=4, n_wx=4, n_rz=2, t=0.0):
    return AirspaceState(
        uav_positions=np.random.randn(n_uav, 3).astype(np.float32) * 100 + 2500,
        uav_velocities=np.random.randn(n_uav, 3).astype(np.float32) * 5,
        uav_headings=np.random.randn(n_uav).astype(np.float32),
        uav_battery=np.ones(n_uav, dtype=np.float32),
        uav_priority=np.ones(n_uav, dtype=np.int32),
        uav_avoiding=np.zeros(n_uav, dtype=bool),
        sector_occupancy=np.zeros((n_sec, 8), dtype=np.float32),
        weather_cells=np.random.randn(n_wx, 12).astype(np.float32) * 100 + 2500,
        restricted_zones=np.array([
            [2500, 2500, 90, 200, 1, 0, 0, 0] for _ in range(n_rz)
        ], dtype=np.float32),
        corridor_reservations=[],
        epoch_time=t,
    )


class TestTKGBuilder:
    def test_build_shape(self):
        builder = TKGBuilder()
        state = _make_state(n_uav=10)
        snapshot = builder.build(state)
        assert snapshot.num_uavs == 10
        assert snapshot.num_nodes == 10 + 4 + 4 + 2
        assert snapshot.node_features.shape == (20, 23)

    def test_edge_types(self):
        builder = TKGBuilder()
        state = _make_state(n_uav=20)
        state.uav_positions[:5] = state.uav_positions[0] + np.random.randn(5, 3) * 5
        snapshot = builder.build(state)
        assert isinstance(snapshot.edge_indices, dict)

    def test_reset(self):
        builder = TKGBuilder()
        state = _make_state()
        builder.build(state)
        builder.reset()
        assert all(len(v) == 0 for v in builder._last_edge_times.values())

    def test_node_types(self):
        builder = TKGBuilder()
        state = _make_state(n_uav=5, n_sec=3, n_wx=2, n_rz=1)
        snapshot = builder.build(state)
        types = snapshot.node_types.numpy()
        assert (types[:5] == 0).all()
        assert (types[5:8] == 1).all()
        assert (types[8:10] == 2).all()
        assert (types[10:] == 3).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
