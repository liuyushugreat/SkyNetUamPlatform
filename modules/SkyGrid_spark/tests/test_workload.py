"""Workload reproducibility and structural tests."""

from __future__ import annotations

from skygrid.config import SkyGridConfig
from skygrid.workload import CityScaleWorkload
from skygrid.workload.dag import TaskDAG


def _small_cfg() -> SkyGridConfig:
    cfg = SkyGridConfig.load(
        # relative path from the tests folder
        __file__.replace("tests", "configs").replace("test_workload.py", "default.yaml")
    )
    cfg.workload.num_entities = 200
    cfg.workload.duration_s = 5.0
    return cfg


def test_workload_is_deterministic():
    cfg = _small_cfg()
    dag = TaskDAG.from_config(cfg.dag)
    w1 = CityScaleWorkload(cfg.workload, dag, seed=cfg.seed).replay()
    w2 = CityScaleWorkload(cfg.workload, dag, seed=cfg.seed).replay()
    assert len(w1) == len(w2)
    for a, b in zip(w1[:200], w2[:200]):
        assert a.event_id == b.event_id
        assert a.eid == b.eid
        assert abs(a.t - b.t) < 1e-9


def test_dag_is_topological():
    cfg = _small_cfg()
    dag = TaskDAG.from_config(cfg.dag)
    order = dag.topo
    position = {n: i for i, n in enumerate(order)}
    for u, v in dag.edges:
        assert position[u] < position[v], f"{u} -> {v} violates topo order"


def test_mobility_stays_in_bounds():
    cfg = _small_cfg()
    dag = TaskDAG.from_config(cfg.dag)
    w = CityScaleWorkload(cfg.workload, dag, seed=cfg.seed)
    xs = w.tracks.xs
    ys = w.tracks.ys
    assert xs.min() >= 0.0
    assert ys.min() >= 0.0
    assert xs.max() <= cfg.workload.area_km + 1e-5
    assert ys.max() <= cfg.workload.area_km + 1e-5
