"""End-to-end smoke test: runtime drives a small workload through the DAG."""

from __future__ import annotations

from skygrid.config import SkyGridConfig
from skygrid.runtime import RuntimeConfig, SkyGridRuntime
from skygrid.workload import CityScaleWorkload
from skygrid.workload.dag import TaskDAG


def _tiny_cfg() -> SkyGridConfig:
    cfg = SkyGridConfig.load(
        __file__.replace("tests", "configs").replace("test_runtime.py", "default.yaml")
    )
    cfg.workload.num_entities = 400
    cfg.workload.duration_s = 4.0
    cfg.fabric.edge.num_nodes = 2
    return cfg


def test_runtime_full_end_to_end():
    cfg = _tiny_cfg()
    dag = TaskDAG.from_config(cfg.dag)
    w = CityScaleWorkload(cfg.workload, dag, seed=cfg.seed)
    rt = SkyGridRuntime(cfg, RuntimeConfig(label="smoke"))
    metrics = rt.run(w)
    assert metrics.num_events > 0
    assert metrics.completed_events > 0
    assert metrics.latency_ms["p95"] >= metrics.latency_ms["p50"]
    assert metrics.latency_ms["p99"] >= metrics.latency_ms["p95"]
    assert metrics.throughput_ops > 0.0


def test_runtime_sync_and_abp_are_stable():
    """Sanity contract: both pipeline modes complete ≥95% of offered events
    and produce bounded tail latencies on a 4 s workload.

    The apples-to-apples throughput advantage of ABP is reported in
    scripts/run_experiment.py (Table 2 / Fig. 4) where steady-state
    windows are compared.  This unit test only asserts that both modes
    are *functionally correct*.
    """
    cfg = _tiny_cfg()
    dag = TaskDAG.from_config(cfg.dag)

    cfg.pipeline.method = "abp"
    w = CityScaleWorkload(cfg.workload, dag, seed=cfg.seed)
    m_abp = SkyGridRuntime(cfg, RuntimeConfig(label="abp")).run(w)

    cfg.pipeline.method = "sync"
    w2 = CityScaleWorkload(cfg.workload, dag, seed=cfg.seed)
    m_sync = SkyGridRuntime(cfg, RuntimeConfig(label="sync")).run(w2)

    for m in (m_abp, m_sync):
        assert m.completed_events >= 0.95 * m.num_events
        assert m.latency_ms["p99"] >= m.latency_ms["p50"]
        assert m.latency_ms["p99"] < 2000.0   # hard bound: 2 s is pathological
