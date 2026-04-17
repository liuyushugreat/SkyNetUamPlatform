"""Inject controlled link-layer disturbance on radar packets (E3 stress)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LinkDisturbance:
    packet_dropout_pct: float = 0.5
    jitter_ms_std: float = 4.0
    auth_delay_ms_mean: float = 18.0
    comm_jitter_ms: float = 1.5


def inject_link_disturbance(
    packets: list,
    rng: np.random.Generator,
    disturbance: LinkDisturbance,
) -> list:
    """Add latency / drop a fraction of incoming radar packets in place.

    Returns the surviving packets; mutates each packet's ``t_recv_ms``.
    """
    surviving = []
    for pkt in packets:
        if rng.random() * 100.0 < disturbance.packet_dropout_pct:
            continue
        pkt.t_recv_ms += abs(rng.normal(disturbance.comm_jitter_ms, disturbance.jitter_ms_std))
        surviving.append(pkt)
    return surviving
