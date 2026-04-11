"""Pre-defined governance task graphs (DAGs) for different operational scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class TaskNodeType(str, Enum):
    AGENT = "agent"
    GATE = "gate"          # conditional branch
    MERGE = "merge"        # join parallel branches
    ESCALATE = "escalate"  # human-in-the-loop


@dataclass
class TaskNode:
    node_id: str
    node_type: TaskNodeType
    agent_name: Optional[str] = None
    next_nodes: List[str] = field(default_factory=list)
    condition: Optional[str] = None  # for GATE nodes: "violation" / "risk" / "safe"


@dataclass
class TaskGraph:
    """Directed acyclic graph defining agent execution order."""

    name: str
    description: str
    nodes: Dict[str, TaskNode] = field(default_factory=dict)
    entry_node: str = ""

    def add_node(self, node: TaskNode) -> "TaskGraph":
        self.nodes[node.node_id] = node
        return self

    def get_entry(self) -> TaskNode:
        return self.nodes[self.entry_node]

    def get_successors(self, node_id: str) -> List[TaskNode]:
        node = self.nodes[node_id]
        return [self.nodes[nid] for nid in node.next_nodes if nid in self.nodes]


# ── Pre-defined task graph: flight approval workflow ──

TASK_FLIGHT_APPROVAL = TaskGraph(
    name="flight_approval",
    description="飞行申请审批：合规检查 → 风险评估 → 解释生成 → 审计",
    entry_node="compliance",
)
TASK_FLIGHT_APPROVAL.add_node(
    TaskNode("compliance", TaskNodeType.AGENT, agent_name="compliance", next_nodes=["gate_compliance"])
).add_node(
    TaskNode("gate_compliance", TaskNodeType.GATE, condition="compliance_verdict", next_nodes=["risk_assessment", "escalate_violation"])
).add_node(
    TaskNode("risk_assessment", TaskNodeType.AGENT, agent_name="risk_assessment", next_nodes=["explanation"])
).add_node(
    TaskNode("explanation", TaskNodeType.AGENT, agent_name="explanation", next_nodes=["audit"])
).add_node(
    TaskNode("audit", TaskNodeType.AGENT, agent_name="audit", next_nodes=["gate_audit"])
).add_node(
    TaskNode("gate_audit", TaskNodeType.GATE, condition="audit_passed", next_nodes=["done", "retry_rag"])
).add_node(
    TaskNode("retry_rag", TaskNodeType.AGENT, agent_name="risk_assessment", next_nodes=["explanation"])
).add_node(
    TaskNode("escalate_violation", TaskNodeType.ESCALATE, next_nodes=[])
).add_node(
    TaskNode("done", TaskNodeType.MERGE, next_nodes=[])
)


# ── Pre-defined task graph: real-time compliance monitoring ──

TASK_REALTIME_MONITOR = TaskGraph(
    name="realtime_monitor",
    description="实时合规监测：合规检查 → 审计 → 告警",
    entry_node="compliance",
)
TASK_REALTIME_MONITOR.add_node(
    TaskNode("compliance", TaskNodeType.AGENT, agent_name="compliance", next_nodes=["audit"])
).add_node(
    TaskNode("audit", TaskNodeType.AGENT, agent_name="audit", next_nodes=["done"])
).add_node(
    TaskNode("done", TaskNodeType.MERGE, next_nodes=[])
)
