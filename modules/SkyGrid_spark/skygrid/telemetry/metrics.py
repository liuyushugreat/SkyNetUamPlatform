"""Result aggregator: produces the JSON rows consumed by ``plot_results``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..utils import percentiles, safe_json
from .tracer import Tracer


@dataclass
class RunMetrics:
    label: str
    duration_s: float
    num_events: int
    completed_events: int
    partition_info: dict[str, Any] = field(default_factory=dict)
    placement_info: dict[str, Any] = field(default_factory=dict)
    pipeline_info: dict[str, Any] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)
    throughput_ops: float = 0.0
    cross_edge_bytes: float = 0.0
    cloud_util: float = 0.0
    edge_util: list[float] = field(default_factory=list)
    rebalance_moves: int = 0
    state_tier_hit_ratio: float = 0.0
    state_tier_avg_ms: float = 0.0
    state_tier_total_ms: float = 0.0

    def to_json(self) -> dict:
        return safe_json(self.__dict__)


def summarize(
    label: str,
    tracer: Tracer,
    duration_s: float,
    num_events: int,
    partition_info: dict,
    placement_info: dict,
    pipeline_info: dict,
    fabric_snapshot: dict,
    rebalance_moves: int = 0,
) -> RunMetrics:
    lat = tracer.event_latencies_ms()
    completed = len(lat)
    pct = percentiles(lat, [0.5, 0.95, 0.99])
    mean = float(np.mean(lat)) if lat else float("nan")
    mx = float(np.max(lat)) if lat else float("nan")
    throughput = completed / duration_s if duration_s > 0 else 0.0
    # Aggregate state tier stats across all edge nodes.
    st_snaps = fabric_snapshot.get("state_tier", [])
    if st_snaps:
        total_hr = sum(s.get("hit_ratio", 0.0) for s in st_snaps) / len(st_snaps)
        total_avg = sum(s.get("avg_access_ms", 0.0) for s in st_snaps) / len(st_snaps)
        total_st_ms = sum(s.get("total_access_ms", 0.0) for s in st_snaps)
    else:
        total_hr, total_avg, total_st_ms = 0.0, 0.0, 0.0

    return RunMetrics(
        label=label,
        duration_s=float(duration_s),
        num_events=int(num_events),
        completed_events=int(completed),
        partition_info=partition_info,
        placement_info=placement_info,
        pipeline_info=pipeline_info,
        latency_ms={
            "mean": mean,
            "p50": pct["p50"],
            "p95": pct["p95"],
            "p99": pct["p99"],
            "max": mx,
        },
        throughput_ops=float(throughput),
        cross_edge_bytes=float(tracer.cross_edge_bytes),
        cloud_util=float(fabric_snapshot.get("cloud_util", 0.0)),
        edge_util=[float(u) for u in fabric_snapshot.get("edge_util", [])],
        rebalance_moves=int(rebalance_moves),
        state_tier_hit_ratio=float(total_hr),
        state_tier_avg_ms=float(total_avg),
        state_tier_total_ms=float(total_st_ms),
    )
