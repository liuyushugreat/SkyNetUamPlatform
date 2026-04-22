"""Top-level discrete-event fabric binding cloud + edges + network."""

from __future__ import annotations

import numpy as np

from ..config import FabricConfig
from .network import Network
from .node import CloudNode, EdgeNode
from .state_tier import StateTierModel


class Fabric:
    def __init__(self, cfg: FabricConfig, seed: int) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.cloud = CloudNode(
            name="cloud",
            tflops=cfg.cloud.tflops,
            queue_capacity=cfg.cloud.queue_capacity,
        )
        self.edges: list[EdgeNode] = [
            EdgeNode(
                name=f"edge-{i}",
                tflops=cfg.edge.tflops,
                queue_capacity=cfg.edge.queue_capacity,
                edge_id=i,
                state_tier=StateTierModel(cfg.state_tier),
            )
            for i in range(cfg.edge.num_nodes)
        ]
        self.network = Network(cfg.network, self.rng)

    # -------------------------------------------------------- lookup

    def site(self, name: str) -> CloudNode | EdgeNode:
        if name == "cloud":
            return self.cloud
        idx = int(name.split("-")[1])
        return self.edges[idx]

    @property
    def site_names(self) -> list[str]:
        return ["cloud"] + [e.name for e in self.edges]

    # -------------------------------------------------------- metrics sink

    def snapshot(self) -> dict:
        return {
            "cloud_util": self.cloud.utilization(),
            "edge_util": [e.utilization() for e in self.edges],
            "cloud_occ": self.cloud.occupancy(),
            "edge_occ": [e.occupancy() for e in self.edges],
            "state_tier": [e.state_snapshot() for e in self.edges],
        }
