"""Closed-form cost model for a per-op placement decision — COP-H (§4.2).

The cost of placing operator ``o`` at site ``s`` when its producer
operator ``p`` is at site ``s'`` is the sum of:

*   **Compute time**    = ``flops(o) / tflops(s)``
*   **Transfer time**   = ``bytes(o) / bw(s', s) + latency(s', s)``
*   **State-access**    = ``state_refs(o) × E[tier_latency]``
*   **Queueing time**   = ``queue_depth(s) / service_rate(s)``

The state-access term is the key addition in COP-H: it captures the
penalty of reading spatial state from the three-tier store (hot → DGX
unified memory, warm → GP Spark NVMe-oF, cold → remote cloud fetch).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import FabricConfig, StateTierConfig
from ..workload.dag import Op, TaskDAG


@dataclass
class OpCost:
    compute_ms: float
    transfer_ms: float
    state_ms: float
    queue_ms: float

    @property
    def total_ms(self) -> float:
        return self.compute_ms + self.transfer_ms + self.state_ms + self.queue_ms


class CostModel:
    """Per-site closed-form cost estimator (COP-H).

    ``sites`` is the list ``["cloud"] + ["edge-0", "edge-1", …]``; given
    an op ``o``, a candidate site ``s`` and the currently-placed site of
    the immediate producer, the method returns an :class:`OpCost`.

    The ``state_tier`` configuration controls the expected latency per
    state reference.  When state tiering is disabled the model degrades
    gracefully to the original COP cost function.
    """

    def __init__(
        self,
        fabric: FabricConfig,
        override_state_tier_enabled: bool | None = None,
    ) -> None:
        self.fabric = fabric
        self.state_tier_cfg: StateTierConfig = fabric.state_tier
        if override_state_tier_enabled is not None:
            # Produce a shallow copy so the outside fabric is not mutated.
            from dataclasses import replace as _replace
            self.state_tier_cfg = _replace(
                fabric.state_tier, enabled=bool(override_state_tier_enabled)
            )
        self.site_names = ["cloud"] + [
            f"edge-{i}" for i in range(fabric.edge.num_nodes)
        ]
        self.tflops = {self.site_names[0]: fabric.cloud.tflops}
        per_node = fabric.edge.tflops_per_node
        for i, s in enumerate(self.site_names[1:]):
            self.tflops[s] = (
                per_node[i] if per_node is not None else fabric.edge.tflops
            )
        self.queue_depth = {s: 0.0 for s in self.site_names}

    # --------------------------------------------------------------- helpers

    def _link(self, src: str, dst: str) -> tuple[float, float]:
        """Return (one-way latency ms, bandwidth Gbps) between sites."""
        net = self.fabric.network
        if src == dst:
            return (0.05, 200.0)                           # intra-node
        if src == "cloud" or dst == "cloud":
            return (net.edge_cloud_latency_ms, net.edge_cloud_bw_gbps)
        return (net.edge_edge_latency_ms, net.edge_edge_bw_gbps)

    def _state_latency_ms(self, op: Op, site: str) -> float:
        """Expected state-access penalty for one invocation of *op* at *site*.

        Only *symbolic* ops that consume spatial state incur this cost.
        Cloud sites always pay ``cold_latency_ms`` per reference because
        the hot/warm tiers are edge-local.
        """
        st = self.state_tier_cfg
        if not st.enabled:
            return 0.0
        state_refs = getattr(op, "state_refs", 0)
        if state_refs <= 0:
            return 0.0
        if site == "cloud":
            return state_refs * st.cold_latency_ms
        h, w = st.hot_hit_rate, st.warm_hit_rate
        per_ref = (
            h * st.hot_latency_ms
            + w * st.warm_latency_ms
            + (1.0 - h - w) * st.cold_latency_ms
        )
        return state_refs * per_ref

    # --------------------------------------------------------------- cost

    def op_cost(self, op: Op, site: str, producer_site: str | None) -> OpCost:
        tflops = self.tflops[site]
        compute_s = op.cost_flops / (tflops * 1e12)
        compute_ms = compute_s * 1e3

        transfer_ms = 0.0
        if producer_site is not None:
            lat, bw = self._link(producer_site, site)
            transfer_s = (op.input_bytes * 8.0) / (bw * 1e9)
            transfer_ms = lat + transfer_s * 1e3

        state_ms = self._state_latency_ms(op, site)

        service_rate = max(1e-6, tflops * 1e12 / max(op.cost_flops, 1.0))
        queue_ms = 1e3 * self.queue_depth[site] / service_rate

        return OpCost(compute_ms, transfer_ms, state_ms, queue_ms)

    def total_cost(self, dag: TaskDAG, placement: dict[str, str]) -> float:
        """End-to-end critical-path cost of a full per-op placement.

        Root ops are modelled as consuming a stream that originates at
        an edge site; if they are placed on the cloud we correctly
        charge the edge→cloud ingress.
        """
        origin_edge = next((s for s in self.site_names if s.startswith("edge")),
                           self.site_names[0])
        finish: dict[str, float] = {}
        for op in dag:
            s = placement[op.name]
            parents = dag.parents(op.name)
            if not parents:
                parent_finish = 0.0
                producer_site = origin_edge
            else:
                parent_finish = max(finish[p] for p in parents)
                slowest = max(parents, key=lambda p: finish[p])
                producer_site = placement[slowest]
            cost = self.op_cost(op, s, producer_site)
            finish[op.name] = parent_finish + cost.total_ms
        return max(finish.values())

    # --------------------------------------------------------------- queue

    def update_queue(self, site: str, delta: float) -> None:
        self.queue_depth[site] = max(0.0, self.queue_depth[site] + delta)
