"""Edge-cloud / edge-edge link model.

Both links are characterized by a mean one-way latency and a bandwidth.
A Gaussian jitter (``jitter_ms``) is added to each transfer so the
simulator can demonstrate the tail-latency pathologies that motivate
ABP.
"""

from __future__ import annotations

import numpy as np

from ..config import NetworkConfig


class Network:
    def __init__(self, cfg: NetworkConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng

    def transfer_ms(self, src: str, dst: str, num_bytes: int) -> float:
        if src == dst:
            return 0.05
        if src == "cloud" or dst == "cloud":
            base = self.cfg.edge_cloud_latency_ms
            bw = self.cfg.edge_cloud_bw_gbps
        else:
            base = self.cfg.edge_edge_latency_ms
            bw = self.cfg.edge_edge_bw_gbps
        serialization_ms = (num_bytes * 8.0) / (bw * 1e9) * 1e3
        jitter = float(self.rng.normal(0.0, self.cfg.jitter_ms))
        return max(0.0, base + serialization_ms + jitter)
