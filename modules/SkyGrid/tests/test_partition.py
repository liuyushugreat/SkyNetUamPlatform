"""STP and baseline partitioner correctness tests."""

from __future__ import annotations

import numpy as np

from skygrid.config import SkyGridConfig
from skygrid.partition import (
    HashPartitioner,
    LDGPartitioner,
    RandomPartitioner,
    SpatioTemporalPartitioner,
    FMRebalancer,
    partition_metrics,
)
from skygrid.workload import CityScaleWorkload
from skygrid.workload.dag import TaskDAG


def _build_entities(n: int = 800):
    cfg = SkyGridConfig.load(
        __file__.replace("tests", "configs").replace("test_partition.py", "default.yaml")
    )
    cfg.workload.num_entities = n
    cfg.workload.duration_s = 3.0
    dag = TaskDAG.from_config(cfg.dag)
    w = CityScaleWorkload(cfg.workload, dag, seed=cfg.seed)
    return cfg, w


def test_hash_is_stable_across_runs():
    _, w = _build_entities(400)
    p1 = HashPartitioner(num_edges=4).assign(w.entities)
    p2 = HashPartitioner(num_edges=4).assign(w.entities)
    assert (p1.assignment == p2.assignment).all()


def test_every_entity_gets_a_partition():
    _, w = _build_entities(400)
    for P in [
        HashPartitioner(num_edges=4),
        RandomPartitioner(num_edges=4),
        LDGPartitioner(num_edges=4),
        SpatioTemporalPartitioner(num_edges=4),
    ]:
        p = P.assign(w.entities)
        assert p.assignment.min() >= 0
        assert p.assignment.max() < 4
        assert int(p.sizes.sum()) == len(w.entities)


def test_stp_reduces_edge_cut_vs_random():
    _, w = _build_entities(800)
    stp = SpatioTemporalPartitioner(num_edges=4).assign(w.entities)
    rnd = RandomPartitioner(num_edges=4, seed=7).assign(w.entities)
    m_stp = partition_metrics(stp, w.entities,
                              cells_per_side=16)
    m_rnd = partition_metrics(rnd, w.entities,
                              cells_per_side=16)
    assert m_stp["edge_cut"] <= m_rnd["edge_cut"] + 1e-9
    # STP's hard-cap is (1 + γ) ≈ 1.4; allow a small 5% slack for
    # dense-cell quantization.
    assert m_stp["load_imbalance"] <= 1.45


def test_stp_beats_ldg_on_spatial_compactness():
    _, w = _build_entities(800)
    stp = SpatioTemporalPartitioner(num_edges=4).assign(w.entities)
    ldg = LDGPartitioner(num_edges=4).assign(w.entities)
    m_stp = partition_metrics(stp, w.entities, cells_per_side=16)
    m_ldg = partition_metrics(ldg, w.entities, cells_per_side=16)
    assert m_stp["spatial_compactness"] <= m_ldg["spatial_compactness"] + 1e-6


def test_rebalancer_only_fires_when_imbalanced():
    _, w = _build_entities(400)
    # Force a skewed partition: everyone on edge 0.
    p = HashPartitioner(num_edges=4).assign(w.entities)
    p.assignment[:] = 0
    p.sizes = np.array([len(w.entities), 0, 0, 0], dtype=np.int32)
    reb = FMRebalancer(trigger_imbalance=1.25, max_moves_per_pass=200)
    assert reb.needs_rebalance(p)
    p2, moves = reb.rebalance(p)
    assert moves > 0
    assert p2.sizes.max() < p.sizes.max()
