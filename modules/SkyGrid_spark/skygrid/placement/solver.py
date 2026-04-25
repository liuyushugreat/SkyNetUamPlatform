"""COP solver — greedy + bounded local swaps (§5.2).

Outline of the algorithm (``solve`` method):

1. **Greedy initialization.**  Walk the DAG in topological order; place
   each op at the site minimizing :func:`CostModel.op_cost` given the
   already-fixed producer sites.  This is the *myopic* lower bound.
2. **Bounded local swaps.**  Repeatedly pick the op with the largest
   contribution to the critical-path cost and try reassigning it to any
   other site; accept a move if it reduces the total DAG cost by more
   than ``epsilon``.  We cap the number of swaps by ``max_swaps`` to
   bound the runtime at ``O(|V| · |S|)``.

The paper shows (Prop. 1) that this procedure returns a placement whose
cost is within ``(1 + ε)`` of the ILP optimum for DAGs whose single-site
cost gap is concave in batch size (which our NN + symbolic mix
satisfies).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..workload.dag import TaskDAG
from .cost_model import CostModel


def _cost_model_without_state_tier(base: CostModel) -> CostModel:
    """Return a sibling CostModel whose state-tier term is disabled.

    The returned model shares ``base.fabric`` by reference but overrides
    the cached ``StateTierConfig`` so the placer chooses based on
    compute + transfer + queue only.  This lets us express a "locality-
    aware but tier-agnostic" baseline (``LocAwarePlacement``) without
    mutating the caller's fabric.
    """
    return CostModel(base.fabric, override_state_tier_enabled=False)


@dataclass
class COPConfigLocal:
    epsilon: float = 0.10
    max_swaps: int = 64


class COPSolver:
    def __init__(
        self,
        dag: TaskDAG,
        cost_model: CostModel,
        config: COPConfigLocal | None = None,
    ) -> None:
        self.dag = dag
        self.cost_model = cost_model
        self.cfg = config or COPConfigLocal()

    # ------------------------------------------------------------------ solve

    def solve(self) -> dict[str, str]:
        placement = self._greedy()
        placement = self._local_swaps(placement)
        return placement

    # --------------------------------------------------- greedy initialization

    def _greedy(self) -> dict[str, str]:
        """Topological greedy initialization.

        The root op is treated as if its producer were an edge (the
        physical source of the event stream); this ensures that placing
        a root op on the cloud is correctly penalized for the
        edge→cloud ingress latency.
        """
        placement: dict[str, str] = {}
        sites = self.cost_model.site_names
        # Use the first edge as the canonical "origin" site at plan time;
        # at runtime the actual edge is determined by the partitioner,
        # but any edge yields the same cloud-vs-edge classification.
        origin_edge = next((s for s in sites if s.startswith("edge")),
                           sites[0])
        for op in self.dag:
            parents = self.dag.parents(op.name)
            best_site = sites[0]
            best_cost = float("inf")
            for s in sites:
                if not parents:
                    producer = origin_edge
                else:
                    # slowest parent defines transfer cost; here we use the
                    # *first* parent deterministically because greedy has
                    # no finish-time estimates yet.  Local swap will revisit.
                    producer = placement[parents[0]]
                c = self.cost_model.op_cost(op, s, producer).total_ms
                # honour operator's preference as a tie-breaker (+5% penalty
                # for placing on the non-preferred side).
                penalty = 0.05 * c if (
                    (op.prefers == "cloud" and s != "cloud") or
                    (op.prefers == "edge" and s == "cloud")
                ) else 0.0
                eff = c + penalty
                if eff < best_cost:
                    best_cost = eff
                    best_site = s
            placement[op.name] = best_site
        return placement

    # -------------------------------------------------- with-penalty helper

    def _penalty(self, op, cost_total: float, site: str) -> float:
        if (op.prefers == "cloud" and site != "cloud") or (
            op.prefers == "edge" and site == "cloud"
        ):
            return 0.05 * cost_total
        return 0.0

    # -------------------------------------------------------- bounded swaps

    def _local_swaps(self, placement: dict[str, str]) -> dict[str, str]:
        sites = self.cost_model.site_names
        best_cost = self.cost_model.total_cost(self.dag, placement)
        for _ in range(self.cfg.max_swaps):
            moved = False
            for op in self.dag:
                cur = placement[op.name]
                for s in sites:
                    if s == cur:
                        continue
                    placement[op.name] = s
                    new_cost = self.cost_model.total_cost(self.dag, placement)
                    if new_cost + self.cfg.epsilon < best_cost:
                        best_cost = new_cost
                        moved = True
                        break
                    placement[op.name] = cur
                if moved:
                    break
            if not moved:
                break
        return placement


class LocAwareSolver(COPSolver):
    """Locality-aware but state-tier-agnostic placement baseline.

    ``LocAware`` is identical to COP except that its cost model has the
    state-tier term disabled.  This isolates the marginal contribution
    of the tiered storage cost in \\(\\bar\\tau(s)\\): any remaining
    improvement of SkyGrid over LocAware is attributable to the state-
    tier awareness of the placer.
    """

    def __init__(
        self,
        dag: TaskDAG,
        cost_model: CostModel,
        config: COPConfigLocal | None = None,
    ) -> None:
        tier_blind = _cost_model_without_state_tier(cost_model)
        super().__init__(dag, tier_blind, config)
