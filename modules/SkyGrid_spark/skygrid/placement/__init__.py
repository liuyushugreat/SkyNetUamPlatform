from .cost_model import CostModel, OpCost
from .solver import COPSolver, LocAwareSolver
from .baselines import (
    StaticPlacement,
    AllCloudPlacement,
    AllEdgePlacement,
    RandomPlacement,
)

__all__ = [
    "CostModel",
    "OpCost",
    "COPSolver",
    "LocAwareSolver",
    "StaticPlacement",
    "AllCloudPlacement",
    "AllEdgePlacement",
    "RandomPlacement",
    "build_placement",
]


def build_placement(name: str, dag, cost_model):
    name = name.lower()
    if name == "cop":
        return COPSolver(dag, cost_model)
    if name in ("loc_aware", "locaware", "loc-aware"):
        return LocAwareSolver(dag, cost_model)
    if name == "static":
        return StaticPlacement(dag)
    if name == "all_cloud":
        return AllCloudPlacement(dag)
    if name == "all_edge":
        return AllEdgePlacement(dag)
    if name == "random":
        return RandomPlacement(dag)
    raise ValueError(f"unknown placement: {name}")
