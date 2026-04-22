"""Hybrid NN + symbolic task DAG.

Each inference job is a DAG of operators (see ``configs/default.yaml``).
The DAG carries structural metadata the runtime needs: operator kind,
flops, input/output bytes, batchability and the preferred placement
hint (edge vs. cloud).  No tensors are actually materialized — the
discrete-event simulator consumes ``flops`` + ``bytes`` and the
placement cost model queries this object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..config import DAGConfig, OpSpec


@dataclass(frozen=True)
class Op:
    name: str
    kind: str                    # "nn" | "symbolic"
    cost_flops: float
    input_bytes: int
    output_bytes: int
    prefers: str                 # "cloud" | "edge"
    batchable: bool
    batch_sweet: int
    state_refs: int = 0          # spatial state lookups per invocation

    @classmethod
    def from_spec(cls, s: OpSpec) -> "Op":
        return cls(
            name=s.name,
            kind=s.kind,
            cost_flops=float(s.cost_flops),
            input_bytes=int(s.input_bytes),
            output_bytes=int(s.output_bytes),
            prefers=s.prefers,
            batchable=bool(s.batchable),
            batch_sweet=int(s.batch_sweet),
            state_refs=int(getattr(s, "state_refs", 0)),
        )


class TaskDAG:
    """Immutable hybrid inference DAG."""

    def __init__(self, ops: list[Op], edges: list[tuple[str, str]]) -> None:
        self.ops: list[Op] = ops
        self.by_name: dict[str, Op] = {o.name: o for o in ops}
        self.edges: list[tuple[str, str]] = edges
        self._children: dict[str, list[str]] = {o.name: [] for o in ops}
        self._parents: dict[str, list[str]] = {o.name: [] for o in ops}
        for u, v in edges:
            self._children[u].append(v)
            self._parents[v].append(u)
        self._topo = self._toposort()

    def _toposort(self) -> list[str]:
        indeg = {n: len(self._parents[n]) for n in self.by_name}
        order: list[str] = [n for n, d in indeg.items() if d == 0]
        out: list[str] = []
        while order:
            n = order.pop(0)
            out.append(n)
            for c in self._children[n]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    order.append(c)
        if len(out) != len(self.by_name):
            raise ValueError("DAG has a cycle")
        return out

    @property
    def topo(self) -> list[str]:
        return list(self._topo)

    def parents(self, name: str) -> list[str]:
        return list(self._parents[name])

    def children(self, name: str) -> list[str]:
        return list(self._children[name])

    def leaves(self) -> list[str]:
        return [n for n, c in self._children.items() if not c]

    def roots(self) -> list[str]:
        return [n for n, p in self._parents.items() if not p]

    def __iter__(self) -> Iterable[Op]:
        for n in self._topo:
            yield self.by_name[n]

    def __len__(self) -> int:
        return len(self.ops)

    @classmethod
    def from_config(cls, cfg: DAGConfig) -> "TaskDAG":
        ops = [Op.from_spec(s) for s in cfg.ops]
        return cls(ops, list(cfg.edges))
