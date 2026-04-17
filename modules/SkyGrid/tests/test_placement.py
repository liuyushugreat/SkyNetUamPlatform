"""COP solver correctness and approximation tests."""

from __future__ import annotations

from skygrid.config import SkyGridConfig
from skygrid.placement import COPSolver
from skygrid.placement.cost_model import CostModel
from skygrid.placement.baselines import (
    AllCloudPlacement,
    AllEdgePlacement,
    StaticPlacement,
)
from skygrid.workload.dag import TaskDAG


def _dag():
    cfg = SkyGridConfig.load(
        __file__.replace("tests", "configs").replace("test_placement.py", "default.yaml")
    )
    return cfg, TaskDAG.from_config(cfg.dag)


def test_cost_model_all_sites_finite():
    cfg, dag = _dag()
    cm = CostModel(cfg.fabric)
    for op in dag:
        for s in cm.site_names:
            c = cm.op_cost(op, s, producer_site=None)
            assert c.total_ms >= 0.0
            assert c.total_ms < 1e4


def test_cop_is_no_worse_than_all_cloud_or_all_edge():
    cfg, dag = _dag()
    cm = CostModel(cfg.fabric)
    best_baseline = min(
        cm.total_cost(dag, AllCloudPlacement(dag).solve()),
        cm.total_cost(dag, AllEdgePlacement(dag).solve()),
        cm.total_cost(dag, StaticPlacement(dag).solve()),
    )
    cop_cost = cm.total_cost(dag, COPSolver(dag, cm).solve())
    assert cop_cost <= best_baseline + 1e-6


def test_cop_assigns_every_op_to_a_known_site():
    cfg, dag = _dag()
    cm = CostModel(cfg.fabric)
    placement = COPSolver(dag, cm).solve()
    for op in dag:
        assert placement[op.name] in cm.site_names, (
            f"{op.name} got placed at unknown site {placement[op.name]}"
        )
    # COP's total cost must not exceed any single-site assignment.
    one_site = {op.name: cm.site_names[0] for op in dag}
    assert cm.total_cost(dag, placement) <= cm.total_cost(dag, one_site) + 1e-6
