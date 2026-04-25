"""SkyGridRuntime — the single integration point.

The runtime owns:

* a :class:`Fabric` — the DES simulator and its compute sites,
* a partitioner, placement solver, and pipeline executor,
* a :class:`Tracer` and a :class:`RunMetrics` aggregator.

Execution model.  The runtime is a *single-process discrete-event
simulator* driven by the event iterator.  The virtual clock advances
with the emission time of arriving events; between two adjacent arrivals
we drain the priority queue of completed micro-batches up to the new
clock.  A final unbounded drain flushes any open batchers when the
input stream is exhausted.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from ..config import SkyGridConfig
from ..partition import build_partitioner, FMRebalancer, partition_metrics
from ..placement import build_placement
from ..placement.cost_model import CostModel
from ..pipeline import build_pipeline
from ..pipeline.micro_batch import MicroBatch, MicroBatcher
from ..simulator.fabric import Fabric
from ..telemetry.tracer import Tracer
from ..telemetry.metrics import RunMetrics, summarize
from ..workload import CityScaleWorkload
from ..workload.dag import TaskDAG


@dataclass
class RuntimeConfig:
    """Per-run knobs orthogonal to the SkyGridConfig YAML."""
    label: str = "skygrid"
    max_events: int = 10_000_000
    verbose: bool = False


@dataclass(order=True)
class _PendingBatch:
    finish_ms: float
    sequence: int
    batch_id: int = field(compare=False)
    op_name: str = field(compare=False)
    site: str = field(compare=False)
    events: list = field(compare=False, default_factory=list)


class SkyGridRuntime:
    def __init__(
        self,
        cfg: SkyGridConfig,
        runtime_cfg: RuntimeConfig | None = None,
    ) -> None:
        self.cfg = cfg
        self.runtime_cfg = runtime_cfg or RuntimeConfig()
        self.dag = TaskDAG.from_config(cfg.dag)
        self.fabric = Fabric(cfg.fabric, seed=cfg.seed)
        self.cost_model = CostModel(cfg.fabric)
        # Capacity weights derived from per-edge TFLOPS (falls back to
        # uniform 1/K in the homogeneous case).  Used by STP to match
        # partition mass to compute capacity and by the rebalancer to
        # evict entities from degraded edges.
        per_node = cfg.fabric.edge.tflops_per_node
        self._capacity_weights = (
            list(per_node) if per_node is not None
            else [cfg.fabric.edge.tflops] * cfg.fabric.edge.num_nodes
        )
        self.partitioner = build_partitioner(
            cfg.partition.method,
            num_edges=cfg.fabric.edge.num_nodes,
            capacity_weights=self._capacity_weights,
        )
        self.pipeline = (
            build_pipeline(cfg.pipeline.method, cfg=cfg.pipeline.abp)
            if cfg.pipeline.method == "abp"
            else build_pipeline(cfg.pipeline.method)
        )
        self.rebalancer = FMRebalancer(
            trigger_imbalance=cfg.partition.rebalance.trigger_imbalance,
        )

        # Populated at run() time
        self._part = None
        self._placement: dict[str, str] = {}
        self._tracer: Tracer | None = None
        self._ready: list[_PendingBatch] = []
        self._seq: int = 0

    # ------------------------------------------------------------------ run

    def run(self, workload: CityScaleWorkload) -> RunMetrics:
        cfg = self.cfg
        self._tracer = Tracer()
        self._ready = []
        heapq.heapify(self._ready)
        self._seq = 0
        self._part = self.partitioner.assign(workload.entities)
        placer = build_placement(cfg.placement.method, self.dag, self.cost_model)
        self._placement = placer.solve()
        rebalance_moves = 0
        last_rebalance_ms = 0.0
        num_events = 0
        last_now_ms = 0.0

        for ev in workload.events():
            if num_events >= self.runtime_cfg.max_events:
                break
            num_events += 1
            now_ms = ev.t * 1e3
            last_now_ms = now_ms
            self._tracer.record_event_start(ev.event_id, now_ms)

            # Tick: first flush any timed-out open batchers.
            for b in list(self._all_batchers().values()):
                mb = b.tick(now_ms)
                if mb is not None:
                    self._dispatch(mb, now_ms)

            # Drain completions up to the current clock.
            self._drain_until(now_ms)

            # Admit into the root op's micro-batcher.  If a root is
            # back-pressured we first drain, then retry.
            for root in self.dag.roots():
                op = self.dag.by_name[root]
                site = self._site_for(ev, op)
                if not self.pipeline.can_admit(site):
                    self._drain_until(now_ms)
                batcher = self.pipeline.batcher_for(op.name, site)
                mb = batcher.add(ev, now_ms)
                if mb is not None:
                    self._dispatch(mb, now_ms)

            # Periodic rebalance (amortized, capacity-aware).
            if (now_ms - last_rebalance_ms) >= (
                cfg.partition.rebalance.amortize_every_s * 1e3
            ):
                if self.rebalancer.needs_rebalance(
                    self._part, self._capacity_weights
                ):
                    new_part, moves = self.rebalancer.rebalance(
                        self._part, self._capacity_weights
                    )
                    rebalance_moves += moves
                    self._part = new_part
                last_rebalance_ms = now_ms

        # Terminal drain: flush all open batchers, then drain to infinity.
        end_ms = max(last_now_ms, cfg.workload.duration_s * 1e3) + 100.0
        for b in list(self._all_batchers().values()):
            mb = b.flush(end_ms)
            if mb is not None:
                self._dispatch(mb, end_ms)
            for m in list(b.drain()):
                self._dispatch(m, end_ms)
        self._drain_until(float("inf"))

        # Utilization snapshot.
        self.fabric.cloud.sample_utilization(end_ms)
        for e in self.fabric.edges:
            e.sample_utilization(end_ms)

        pm = partition_metrics(
            self._part, workload.entities,
            cells_per_side=cfg.partition.grid.cells_per_side,
        )
        return summarize(
            label=self.runtime_cfg.label,
            tracer=self._tracer,
            duration_s=cfg.workload.duration_s,
            num_events=num_events,
            partition_info={"method": cfg.partition.method, **pm},
            placement_info={"method": cfg.placement.method,
                            "assignment": self._placement},
            pipeline_info={"method": cfg.pipeline.method},
            fabric_snapshot=self.fabric.snapshot(),
            rebalance_moves=rebalance_moves,
        )

    # ------------------------------------------------------------------ helpers

    def _all_batchers(self) -> dict[tuple[str, str], MicroBatcher]:
        assert hasattr(self.pipeline, "_batchers")
        return self.pipeline._batchers  # type: ignore[attr-defined]

    def _site_for(self, event, op) -> str:
        """Map an event + op to its physical site via placement + partition."""
        target = self._placement[op.name]
        if target.startswith("edge"):
            ed = int(self._part.of(event.eid))
            return f"edge-{ed}"
        return target

    # --------------------------------------------------------- _dispatch

    def _dispatch(self, mb: MicroBatch, now_ms: float) -> None:
        if not mb.events:
            return
        assert self._tracer is not None
        op = self.dag.by_name[mb.op_name]
        node = self.fabric.site(mb.site)
        node.admit(len(mb.events))
        self.pipeline.mark_dispatch(mb.site)

        # Per-event transfer cost; cross-edge bytes for cross-site hops.
        #
        # Root ops incur an ingress transfer from the event's origin edge
        # (decided by the partitioner) to the placement site.  Non-root ops
        # incur a transfer from their parent's site (stored in _last_site).
        arrival_ms = max(now_ms, mb.flushed_at_ms or now_ms)
        parents = self.dag.parents(op.name)
        if parents:
            parent_op = self.dag.by_name[parents[0]]
            payload_bytes = parent_op.output_bytes
        else:
            parent_op = None
            payload_bytes = op.input_bytes
        max_tms = 0.0
        # For ``rule_check`` we additionally pull peer features from every
        # spatially-nearby entity (``event.peer_eids``).  Those peer rows
        # live on the edge that owns their home cell, so cross-edge
        # traffic is the bytes per peer times the fraction of peers on a
        # different partition — the quantity that STP directly
        # minimizes.
        peer_bytes = op.input_bytes if op.name == "rule_check" else 0
        for ev in mb.events:
            if parents:
                src = getattr(ev, "_last_site", mb.site)
            else:
                ed = int(self._part.of(ev.eid)) if self._part is not None else 0
                src = f"edge-{ed}"
            tms = self.fabric.network.transfer_ms(src, mb.site, payload_bytes)
            if src != mb.site:
                self._tracer.add_cross_edge_bytes(payload_bytes)
            if peer_bytes and self._part is not None:
                for peer in getattr(ev, "peer_eids", ()):
                    ped = int(self._part.of(peer))
                    psrc = f"edge-{ped}"
                    ptms = self.fabric.network.transfer_ms(
                        psrc, mb.site, peer_bytes,
                    )
                    if psrc != mb.site:
                        self._tracer.add_cross_edge_bytes(peer_bytes)
                    if ptms > tms:
                        tms = ptms
            if tms > max_tms:
                max_tms = tms
        arrival_ms = arrival_ms + max_tms

        # State-access penalty: symbolic ops reading spatial state incur a
        # tier-dependent latency (hot → unified memory, warm → NVMe, cold → cloud).
        from ..simulator.node import EdgeNode as _EN
        if isinstance(node, _EN):
            arrival_ms += node.state_access_ms(op, batch_size=len(mb.events))

        finish_ms = node.enqueue(arrival_ms, op, batch_size=len(mb.events))
        node.sample_utilization(finish_ms)
        self.pipeline.observe_occupancy(mb.site, node.occupancy())

        for ev in mb.events:
            self._tracer.record_span(ev.event_id, mb.op_name, mb.site,
                                     arrival_ms, finish_ms)

        self._seq += 1
        heapq.heappush(self._ready, _PendingBatch(
            finish_ms=finish_ms,
            sequence=self._seq,
            batch_id=mb.batch_id,
            op_name=mb.op_name,
            site=mb.site,
            events=list(mb.events),
        ))

    # --------------------------------------------------------- _drain_until

    def _drain_until(self, horizon_ms: float) -> None:
        """Drain completions up to ``horizon_ms`` and cascade downstream.

        After each wave of completions we also tick every open batcher
        using the latest virtual clock so that timed-out micro-batches
        are flushed in simulation time (not wall-clock time).  This is
        what enforces the ABP ``microbatch_timeout_ms`` bound.
        """
        assert self._tracer is not None
        made_progress = True
        while made_progress:
            made_progress = False
            # 1) Pop every completed batch whose finish_ms is due.
            while self._ready and self._ready[0].finish_ms <= horizon_ms:
                pb = heapq.heappop(self._ready)
                self.pipeline.mark_complete(pb.site)
                children = self.dag.children(pb.op_name)
                for ev in pb.events:
                    ev._last_site = pb.site  # type: ignore[attr-defined]
                    if not children:
                        self._tracer.record_event_end(ev.event_id, pb.finish_ms)
                        continue
                    # Join semantics: each event fires every downstream op at
                    # most once.  When a DAG diamond reconverges (e.g. both
                    # feat_extract and risk_score feed rule_check) we want
                    # rule_check to run exactly once per event, not once per
                    # incoming edge; otherwise we would double-count compute
                    # and mis-attribute latency.  The set is attached to the
                    # event so the bookkeeping is O(1) and process-local.
                    triggered: set[str] = getattr(ev, "_triggered_ops", None) or set()
                    ev._triggered_ops = triggered  # type: ignore[attr-defined]
                    for child in children:
                        if child in triggered:
                            continue
                        triggered.add(child)
                        child_op = self.dag.by_name[child]
                        site = self._site_for(ev, child_op)
                        batcher = self.pipeline.batcher_for(child_op.name, site)
                        mb = batcher.add(ev, pb.finish_ms)
                        if mb is not None:
                            self._dispatch(mb, pb.finish_ms)
                made_progress = True

            # 2) Timeout-flush any open batchers whose oldest event is
            #    older than the configured timeout under the virtual clock.
            cap_ms = horizon_ms if horizon_ms != float("inf") else (
                self._ready[0].finish_ms if self._ready else 0.0
            )
            for b in list(self._all_batchers().values()):
                mb = b.tick(cap_ms)
                if mb is not None:
                    self._dispatch(mb, cap_ms)
                    made_progress = True
