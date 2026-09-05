"""Domain-independent runtime contract used by portability experiments.

The module keeps candidate admission, externally visible effect commitment,
receipt reconciliation, and impact-closure repair independent of any UAV or
DevOps vocabulary.  Domain adapters provide parsing, grounding, binding,
repair patches, and invariant checks.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol


CORE_CONTRACT_COMPONENTS = (
    "typed_candidate_admission",
    "proposal_adjudication_commit",
    "idempotency_and_receipts",
    "effect_reconciliation",
    "causal_impact_closure",
    "closure_external_commitment_locking",
    "post_repair_invariant_check",
)


class ContractState(str, Enum):
    PROPOSED = "Proposed"
    PRECHECKED = "Prechecked"
    EXECUTING = "Executing"
    EFFECT_UNKNOWN = "EffectUnknown"
    COMMITTED = "Committed"
    COMPENSATING = "Compensating"
    RECOVERED = "Recovered"
    REJECTED = "Rejected"
    HUMAN_ESCALATED = "HumanEscalated"


@dataclass(frozen=True)
class StructuredFailure:
    code: str
    message: str


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    action: str
    target: str
    resource: str
    slot: str
    permission: str
    requires_approval: bool = False
    approval_granted: bool = False
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowNode:
    node_id: str
    kind: str
    state: ContractState
    binding: dict[str, Any]
    committed: bool = True


@dataclass
class Workflow:
    workflow_id: str
    nodes: dict[str, WorkflowNode]
    causal_edges: dict[str, set[str]]
    version: int = 1
    processed_event_ids: set[str] = field(default_factory=set)
    operation_states: dict[str, ContractState] = field(default_factory=dict)
    receipts: dict[str, "ExecutionReceipt"] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class RuntimeEvent:
    event_id: str
    workflow_id: str
    event_type: str
    directly_affected: frozenset[str]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionReceipt:
    idempotency_key: str
    receipt_id: str
    workflow_version: int
    causal_parent: str


@dataclass(frozen=True)
class AdmissionResult:
    case_id: str
    executable: bool
    workflow: Workflow | None = None
    failure: StructuredFailure | None = None


@dataclass(frozen=True)
class CommitResult:
    state: ContractState
    invoke_count: int
    effect_count: int
    receipt_count: int
    reconciled: bool


@dataclass(frozen=True)
class EventResult:
    event_id: str
    workflow_id: str
    status: str
    impact_closure: tuple[str, ...]
    changed_nodes: int
    total_nodes: int
    protected_commitments: int
    preserved_commitments: int
    commit_result: CommitResult | None = None
    failure: StructuredFailure | None = None

    @property
    def change_ratio(self) -> float:
        return self.changed_nodes / self.total_nodes if self.total_nodes else 0.0

    @property
    def commitment_preservation(self) -> float:
        if not self.protected_commitments:
            return 1.0
        return self.preserved_commitments / self.protected_commitments


class ContractAdapter(Protocol):
    """Operations that must be replaced for a new application domain."""

    domain_name: str
    replaced_components: tuple[str, ...]

    def parse_instruction(self, case_id: str, instruction: str) -> Candidate | StructuredFailure:
        ...

    def adjudicate(self, candidate: Candidate) -> StructuredFailure | None:
        ...

    def build_workflow(self, candidate: Candidate) -> Workflow:
        ...

    def plan_repair(
        self,
        workflow: Workflow,
        event: RuntimeEvent,
        impact_closure: frozenset[str],
    ) -> dict[str, dict[str, Any]] | StructuredFailure:
        ...

    def check_invariants(self, workflow: Workflow) -> StructuredFailure | None:
        ...


class IdempotentReceiver:
    """Deterministic receiving end with key-based effect deduplication."""

    def __init__(self) -> None:
        self._invoke_counts: dict[str, int] = {}
        self._effect_counts: dict[str, int] = {}
        self._receipts: dict[str, ExecutionReceipt] = {}

    def invoke(
        self,
        idempotency_key: str,
        *,
        workflow_version: int,
        causal_parent: str,
        payload: Mapping[str, Any],
    ) -> ExecutionReceipt:
        self._invoke_counts[idempotency_key] = self._invoke_counts.get(idempotency_key, 0) + 1
        if idempotency_key not in self._receipts:
            encoded = json.dumps(
                {
                    "key": idempotency_key,
                    "version": workflow_version,
                    "causal_parent": causal_parent,
                    "payload": dict(payload),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            receipt = ExecutionReceipt(
                idempotency_key=idempotency_key,
                receipt_id=hashlib.sha256(encoded.encode()).hexdigest()[:20],
                workflow_version=workflow_version,
                causal_parent=causal_parent,
            )
            self._receipts[idempotency_key] = receipt
            self._effect_counts[idempotency_key] = 1
        return self._receipts[idempotency_key]

    def query(self, idempotency_key: str) -> ExecutionReceipt | None:
        return self._receipts.get(idempotency_key)

    def invoke_count(self, idempotency_key: str) -> int:
        return self._invoke_counts.get(idempotency_key, 0)

    def effect_count(self, idempotency_key: str) -> int:
        return self._effect_counts.get(idempotency_key, 0)

    @property
    def total_invocations(self) -> int:
        return sum(self._invoke_counts.values())

    @property
    def total_effects(self) -> int:
        return sum(self._effect_counts.values())

    @property
    def duplicate_effects(self) -> int:
        return sum(max(0, count - 1) for count in self._effect_counts.values())

    @property
    def duplicate_invocations(self) -> int:
        return sum(max(0, count - 1) for count in self._invoke_counts.values())


class RuntimeContract:
    """Execute the same runtime obligations for any ``ContractAdapter``."""

    def __init__(self, adapter: ContractAdapter, receiver: IdempotentReceiver | None = None) -> None:
        self.adapter = adapter
        self.receiver = receiver or IdempotentReceiver()

    def admit(self, case_id: str, instruction: str) -> AdmissionResult:
        parsed = self.adapter.parse_instruction(case_id, instruction)
        if isinstance(parsed, StructuredFailure):
            return AdmissionResult(case_id=case_id, executable=False, failure=parsed)
        failure = self.adapter.adjudicate(parsed)
        if failure is not None:
            return AdmissionResult(case_id=case_id, executable=False, failure=failure)
        workflow = self.adapter.build_workflow(parsed)
        failure = self.adapter.check_invariants(workflow)
        if failure is not None:
            return AdmissionResult(case_id=case_id, executable=False, failure=failure)
        workflow.evidence.append({"kind": "candidate_admitted", "case_id": case_id})
        return AdmissionResult(case_id=case_id, executable=True, workflow=workflow)

    @staticmethod
    def impact_closure(workflow: Workflow, directly_affected: frozenset[str]) -> frozenset[str]:
        unknown = set(directly_affected) - set(workflow.nodes)
        if unknown:
            raise KeyError(f"Unknown directly affected nodes: {sorted(unknown)}")
        closure = set(directly_affected)
        pending = list(directly_affected)
        while pending:
            source = pending.pop()
            for dependent in workflow.causal_edges.get(source, set()):
                if dependent not in closure:
                    closure.add(dependent)
                    pending.append(dependent)
        return frozenset(closure)

    def commit_external_effect(
        self,
        workflow: Workflow,
        *,
        idempotency_key: str,
        causal_parent: str,
        payload: Mapping[str, Any],
        node_id: str | None = None,
        simulate_receipt_loss: bool = False,
        receiver_query_available: bool = True,
    ) -> CommitResult:
        if node_id is not None and node_id not in workflow.nodes:
            raise KeyError(f"Unknown external-action node: {node_id}")
        current = workflow.operation_states.get(idempotency_key, ContractState.PRECHECKED)
        if current == ContractState.COMMITTED:
            return CommitResult(
                state=current,
                invoke_count=self.receiver.invoke_count(idempotency_key),
                effect_count=self.receiver.effect_count(idempotency_key),
                receipt_count=int(idempotency_key in workflow.receipts),
                reconciled=False,
            )
        if current not in {ContractState.PRECHECKED, ContractState.EFFECT_UNKNOWN}:
            raise RuntimeError(f"Cannot issue an invocation from {current.value}")

        reconciled = current == ContractState.EFFECT_UNKNOWN
        receipt: ExecutionReceipt | None = None
        if current == ContractState.PRECHECKED:
            workflow.operation_states[idempotency_key] = ContractState.EXECUTING
            if node_id is not None:
                workflow.nodes[node_id].state = ContractState.EXECUTING
                workflow.nodes[node_id].committed = False
            workflow.evidence.append(
                {"kind": "invocation_issued", "key": idempotency_key, "parent": causal_parent}
            )
            receipt = self.receiver.invoke(
                idempotency_key,
                workflow_version=workflow.version,
                causal_parent=causal_parent,
                payload=payload,
            )
            if simulate_receipt_loss:
                workflow.operation_states[idempotency_key] = ContractState.EFFECT_UNKNOWN
                if node_id is not None:
                    workflow.nodes[node_id].state = ContractState.EFFECT_UNKNOWN
                workflow.evidence.append({"kind": "effect_outcome_unknown", "key": idempotency_key})
                receipt = None

        if workflow.operation_states[idempotency_key] == ContractState.EFFECT_UNKNOWN:
            if receiver_query_available:
                receipt = self.receiver.query(idempotency_key)
                reconciled = True
            else:
                workflow.operation_states[idempotency_key] = ContractState.HUMAN_ESCALATED
                if node_id is not None:
                    workflow.nodes[node_id].state = ContractState.HUMAN_ESCALATED
                workflow.evidence.append({"kind": "reconciliation_unavailable", "key": idempotency_key})
                return CommitResult(
                    state=ContractState.HUMAN_ESCALATED,
                    invoke_count=self.receiver.invoke_count(idempotency_key),
                    effect_count=self.receiver.effect_count(idempotency_key),
                    receipt_count=0,
                    reconciled=False,
                )

        if receipt is None:
            workflow.operation_states[idempotency_key] = ContractState.PRECHECKED
            if node_id is not None:
                workflow.nodes[node_id].state = ContractState.PRECHECKED
            workflow.evidence.append({"kind": "reconciled_effect_absent", "key": idempotency_key})
            return CommitResult(
                state=ContractState.PRECHECKED,
                invoke_count=self.receiver.invoke_count(idempotency_key),
                effect_count=self.receiver.effect_count(idempotency_key),
                receipt_count=0,
                reconciled=reconciled,
            )

        if receipt.workflow_version != workflow.version or receipt.causal_parent != causal_parent:
            raise RuntimeError("Receipt does not match workflow version and causal parent")
        workflow.receipts[idempotency_key] = receipt
        workflow.operation_states[idempotency_key] = ContractState.COMMITTED
        if node_id is not None:
            workflow.nodes[node_id].state = ContractState.COMMITTED
            workflow.nodes[node_id].committed = True
        workflow.evidence.append(
            {"kind": "receipt_reconciled" if reconciled else "receipt_persisted", "key": idempotency_key}
        )
        return CommitResult(
            state=ContractState.COMMITTED,
            invoke_count=self.receiver.invoke_count(idempotency_key),
            effect_count=self.receiver.effect_count(idempotency_key),
            receipt_count=1,
            reconciled=reconciled,
        )

    def process_event(self, workflow: Workflow, event: RuntimeEvent) -> EventResult:
        if event.workflow_id != workflow.workflow_id:
            raise ValueError("Event and workflow identifiers do not match")
        if event.event_id in workflow.processed_event_ids:
            return EventResult(
                event_id=event.event_id,
                workflow_id=workflow.workflow_id,
                status="duplicate_ignored",
                impact_closure=(),
                changed_nodes=0,
                total_nodes=len(workflow.nodes),
                protected_commitments=sum(node.committed for node in workflow.nodes.values()),
                preserved_commitments=sum(node.committed for node in workflow.nodes.values()),
            )

        closure = self.impact_closure(workflow, event.directly_affected)
        protected_before = {
            node_id: (node.state, copy.deepcopy(node.binding))
            for node_id, node in workflow.nodes.items()
            if node_id not in closure and node.committed
        }
        before = {
            node_id: (node.state, copy.deepcopy(node.binding))
            for node_id, node in workflow.nodes.items()
        }

        patches = self.adapter.plan_repair(workflow, event, closure)
        if isinstance(patches, StructuredFailure):
            workflow.processed_event_ids.add(event.event_id)
            workflow.evidence.append(
                {
                    "kind": "event_escalated",
                    "event_id": event.event_id,
                    "type": event.event_type,
                    "failure": patches.code,
                }
            )
            return EventResult(
                event_id=event.event_id,
                workflow_id=workflow.workflow_id,
                status="escalated",
                impact_closure=tuple(sorted(closure)),
                changed_nodes=0,
                total_nodes=len(workflow.nodes),
                protected_commitments=len(protected_before),
                preserved_commitments=len(protected_before),
                failure=patches,
            )
        outside = set(patches) - set(closure)
        if outside:
            raise RuntimeError(f"Adapter attempted to patch nodes outside the impact closure: {sorted(outside)}")

        proposed = copy.deepcopy(workflow)
        for node_id in closure:
            proposed.nodes[node_id].state = ContractState.COMPENSATING
        for node_id, binding_patch in patches.items():
            proposed.nodes[node_id].binding.update(binding_patch)
        for node_id in closure:
            proposed.nodes[node_id].state = ContractState.RECOVERED
        proposed.version = workflow.version + 1

        failure = self.adapter.check_invariants(proposed)
        if failure is not None:
            workflow.processed_event_ids.add(event.event_id)
            return EventResult(
                event_id=event.event_id,
                workflow_id=workflow.workflow_id,
                status="escalated",
                impact_closure=tuple(sorted(closure)),
                changed_nodes=0,
                total_nodes=len(workflow.nodes),
                protected_commitments=len(protected_before),
                preserved_commitments=len(protected_before),
                failure=failure,
            )

        for node_id, snapshot in protected_before.items():
            candidate = proposed.nodes[node_id]
            if (candidate.state, candidate.binding) != snapshot:
                raise RuntimeError(f"Protected commitment changed outside closure: {node_id}")

        idempotency_key = str(
            event.metadata.get(
                "idempotency_key",
                f"{workflow.workflow_id}:{event.event_id}:v{proposed.version}",
            )
        )
        commit = self.commit_external_effect(
            proposed,
            idempotency_key=idempotency_key,
            causal_parent=event.event_id,
            payload={"event_type": event.event_type, "patches": patches},
            node_id=str(event.metadata.get("effect_node", "external_action")),
            simulate_receipt_loss=event.event_type == "receipt_missing",
        )
        if commit.state != ContractState.COMMITTED:
            raise RuntimeError(f"Recoverable event did not reach Committed: {commit.state.value}")

        proposed.processed_event_ids.add(event.event_id)
        proposed.evidence.append(
            {"kind": "repair_committed", "event_id": event.event_id, "closure": sorted(closure)}
        )
        changed = sum(
            before[node_id] != (proposed.nodes[node_id].state, proposed.nodes[node_id].binding)
            for node_id in proposed.nodes
        )
        preserved = sum(
            (proposed.nodes[node_id].state, proposed.nodes[node_id].binding) == snapshot
            for node_id, snapshot in protected_before.items()
        )
        workflow.nodes = proposed.nodes
        workflow.causal_edges = proposed.causal_edges
        workflow.version = proposed.version
        workflow.processed_event_ids = proposed.processed_event_ids
        workflow.operation_states = proposed.operation_states
        workflow.receipts = proposed.receipts
        workflow.evidence = proposed.evidence
        return EventResult(
            event_id=event.event_id,
            workflow_id=workflow.workflow_id,
            status="repaired",
            impact_closure=tuple(sorted(closure)),
            changed_nodes=changed,
            total_nodes=len(workflow.nodes),
            protected_commitments=len(protected_before),
            preserved_commitments=preserved,
            commit_result=commit,
        )
