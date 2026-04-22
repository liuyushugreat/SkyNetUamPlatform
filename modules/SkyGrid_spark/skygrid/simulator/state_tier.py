"""Three-tier state access model: hot (DGX memory) / warm (GP Spark) / cold (cloud).

Each edge unit maintains a tiered state store that models the latency
penalty of reading spatial neighbour features, rule snapshots, and
audit state.  The tier hit/miss ratios determine the effective state
access cost that feeds into the COP-H placement cost model.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import StateTierConfig


@dataclass
class StateTierStats:
    """Running counters for a single edge node's state tier."""

    hot_hits: int = 0
    warm_hits: int = 0
    cold_hits: int = 0
    total_access_ms: float = 0.0

    @property
    def total_accesses(self) -> int:
        return self.hot_hits + self.warm_hits + self.cold_hits

    @property
    def hit_ratio(self) -> float:
        t = self.total_accesses
        return (self.hot_hits + self.warm_hits) / t if t > 0 else 1.0

    @property
    def avg_access_ms(self) -> float:
        t = self.total_accesses
        return self.total_access_ms / t if t > 0 else 0.0


class StateTierModel:
    """Per-edge-node state tier simulator.

    On each state access the tier is selected using the configured hit
    rates (expected-value model — deterministic, no RNG needed).
    The returned latency feeds into COP-H's state-access cost term and
    into the runtime's dispatch path.
    """

    def __init__(self, cfg: StateTierConfig | None = None) -> None:
        self.cfg = cfg or StateTierConfig()
        self.stats = StateTierStats()

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    def access_latency_ms(self, num_refs: int = 1) -> float:
        """Return the aggregate state-access latency for *num_refs* lookups."""
        if not self.cfg.enabled or num_refs <= 0:
            return 0.0

        h = self.cfg.hot_hit_rate
        w = self.cfg.warm_hit_rate
        c = 1.0 - h - w

        per_ref = (
            h * self.cfg.hot_latency_ms
            + w * self.cfg.warm_latency_ms
            + c * self.cfg.cold_latency_ms
        )
        total_ms = per_ref * num_refs

        hot_n = int(num_refs * h)
        warm_n = int(num_refs * w)
        cold_n = num_refs - hot_n - warm_n
        self.stats.hot_hits += hot_n
        self.stats.warm_hits += warm_n
        self.stats.cold_hits += cold_n
        self.stats.total_access_ms += total_ms
        return total_ms

    def expected_latency_ms(self, num_refs: int = 1) -> float:
        """Pure estimate without updating stats (used by cost model)."""
        if not self.cfg.enabled or num_refs <= 0:
            return 0.0
        h = self.cfg.hot_hit_rate
        w = self.cfg.warm_hit_rate
        per_ref = (
            h * self.cfg.hot_latency_ms
            + w * self.cfg.warm_latency_ms
            + (1.0 - h - w) * self.cfg.cold_latency_ms
        )
        return per_ref * num_refs

    def snapshot(self) -> dict:
        return {
            "hot_hits": self.stats.hot_hits,
            "warm_hits": self.stats.warm_hits,
            "cold_hits": self.stats.cold_hits,
            "hit_ratio": round(self.stats.hit_ratio, 4),
            "avg_access_ms": round(self.stats.avg_access_ms, 4),
            "total_access_ms": round(self.stats.total_access_ms, 3),
        }

    def reset(self) -> None:
        self.stats = StateTierStats()
