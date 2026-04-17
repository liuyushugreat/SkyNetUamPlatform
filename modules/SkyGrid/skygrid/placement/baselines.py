"""Baseline placement strategies."""

from __future__ import annotations

from ..workload.dag import TaskDAG
from ..utils import make_rng


class StaticPlacement:
    """NN operators on the cloud, symbolic operators on the edge.

    The edge is always ``edge-0``; this is the naive baseline the paper
    calls ``static``.  A more fair "preferred" variant honours the
    operator's ``prefers`` hint.
    """

    def __init__(self, dag: TaskDAG, edge_name: str = "edge-0") -> None:
        self.dag = dag
        self.edge_name = edge_name

    def solve(self) -> dict[str, str]:
        placement: dict[str, str] = {}
        for op in self.dag:
            if op.prefers == "cloud" or op.kind == "nn":
                placement[op.name] = "cloud"
            else:
                placement[op.name] = self.edge_name
        return placement


class AllCloudPlacement:
    def __init__(self, dag: TaskDAG) -> None:
        self.dag = dag

    def solve(self) -> dict[str, str]:
        return {op.name: "cloud" for op in self.dag}


class AllEdgePlacement:
    def __init__(self, dag: TaskDAG, edge_name: str = "edge-0") -> None:
        self.dag = dag
        self.edge_name = edge_name

    def solve(self) -> dict[str, str]:
        return {op.name: self.edge_name for op in self.dag}


class RandomPlacement:
    def __init__(self, dag: TaskDAG, seed: int = 12345,
                 sites: list[str] | None = None) -> None:
        self.dag = dag
        self.seed = int(seed)
        self.sites = sites

    def solve(self) -> dict[str, str]:
        rng = make_rng(self.seed)
        sites = self.sites or ["cloud", "edge-0"]
        return {op.name: sites[int(rng.integers(0, len(sites)))]
                for op in self.dag}
