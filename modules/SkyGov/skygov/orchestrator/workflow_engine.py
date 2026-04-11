"""DAG-based multi-agent workflow engine with timeout and retry support."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from ..agents.base_agent import BaseAgent, AgentResult, AgentVerdict
from .task_graph import TaskGraph, TaskNodeType
from .trust_protocol import TrustProtocol

logger = logging.getLogger(__name__)


class WorkflowEngine:
    """Executes a TaskGraph by dispatching to registered agents.

    Handles:
        - Sequential DAG traversal with conditional branching
        - Veto short-circuit (ComplianceAgent can halt the pipeline)
        - Audit-triggered re-retrieval (up to max_retries)
        - Timeout enforcement per workflow run
    """

    def __init__(
        self,
        agents: Dict[str, BaseAgent],
        trust_protocol: Optional[TrustProtocol] = None,
        max_retries: int = 2,
        timeout_seconds: float = 60.0,
    ):
        self.agents = agents
        self.trust = trust_protocol or TrustProtocol()
        self.max_retries = max_retries
        self.timeout = timeout_seconds

    def run(self, task_graph: TaskGraph, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the full governance workflow.

        Args:
            task_graph: DAG defining agent execution order.
            context: Flight scenario data.

        Returns:
            Final governance decision with all intermediate results.
        """
        t0 = time.perf_counter()
        results: Dict[str, AgentResult] = {}
        retries = 0
        current_id = task_graph.entry_node
        pre_retry_audits = []

        while current_id and current_id in task_graph.nodes:
            if time.perf_counter() - t0 > self.timeout:
                logger.error("Workflow timeout after %.1fs", self.timeout)
                break

            node = task_graph.nodes[current_id]
            logger.info("→ Node: %s (type=%s)", node.node_id, node.node_type.value)

            if node.node_type == TaskNodeType.AGENT:
                agent = self.agents.get(node.agent_name)
                if agent is None:
                    logger.error("Agent '%s' not registered, skipping", node.agent_name)
                    current_id = node.next_nodes[0] if node.next_nodes else None
                    continue

                enriched = self._enrich_context(context, results)
                result = agent._timed_execute(enriched)
                results[node.agent_name] = result

                if result.is_blocking:
                    logger.warning("Blocking verdict from %s, short-circuiting", node.agent_name)
                    break

                current_id = node.next_nodes[0] if node.next_nodes else None

            elif node.node_type == TaskNodeType.GATE:
                next_id = self._evaluate_gate(node, results, retries)
                if next_id == "retry_rag":
                    retries += 1
                    audit = results.get("audit")
                    if audit:
                        pre_retry_audits.append({
                            "retry_num": retries,
                            "rar": audit.payload.get("rar"),
                            "lec": audit.payload.get("lec"),
                            "ucr": audit.payload.get("ucr"),
                            "passed": audit.payload.get("passed"),
                        })
                current_id = next_id

            elif node.node_type == TaskNodeType.ESCALATE:
                logger.warning("Escalating to human operator")
                results["_escalation"] = AgentResult(
                    agent_name="system",
                    verdict=AgentVerdict.VETO,
                    payload={"reason": "human_escalation_required"},
                )
                break

            elif node.node_type == TaskNodeType.MERGE:
                current_id = None

            else:
                current_id = node.next_nodes[0] if node.next_nodes else None

        final_decision = self.trust.aggregate(results)
        elapsed = (time.perf_counter() - t0) * 1000

        return {
            "decision": final_decision,
            "agent_results": {k: _result_to_dict(v) for k, v in results.items() if isinstance(v, AgentResult)},
            "total_latency_ms": round(elapsed, 1),
            "retries": retries,
            "pre_retry_audits": pre_retry_audits,
        }

    def _evaluate_gate(self, node, results, retries) -> Optional[str]:
        """Route to the appropriate successor based on gate condition."""
        if node.condition == "compliance_verdict":
            comp = results.get("compliance")
            if comp and comp.verdict == AgentVerdict.VIOLATION:
                return "escalate_violation" if "escalate_violation" in node.next_nodes else None
            return node.next_nodes[0] if node.next_nodes else None

        if node.condition == "audit_passed":
            audit = results.get("audit")
            if audit and audit.payload.get("passed", False):
                return "done" if "done" in node.next_nodes else node.next_nodes[0]
            if retries < self.max_retries:
                return "retry_rag" if "retry_rag" in node.next_nodes else None
            return "done" if "done" in node.next_nodes else None

        return node.next_nodes[0] if node.next_nodes else None

    def _enrich_context(
        self, base_context: Dict[str, Any], results: Dict[str, AgentResult]
    ) -> Dict[str, Any]:
        """Merge prior agent outputs into the context for downstream agents."""
        enriched = dict(base_context)
        comp = results.get("compliance")
        if comp:
            enriched["compliance_result"] = {
                "verdict": comp.verdict.value,
                "traces": [
                    {"step": t.step, "rule_ids": t.rule_ids, "detail": t.detail}
                    for t in comp.traces
                ],
            }
        risk = results.get("risk_assessment")
        if risk:
            enriched["risk_result"] = risk.payload
        expl = results.get("explanation")
        if expl:
            enriched["explanation"] = expl.payload.get("explanation", "")
            enriched["cited_rules"] = expl.payload.get("cited_rules", [])
        audit = results.get("audit")
        if audit:
            enriched["audit_feedback"] = {
                "rar": audit.payload.get("rar"),
                "ucr": audit.payload.get("ucr"),
                "lec": audit.payload.get("lec"),
                "passed": audit.payload.get("passed"),
            }
            enriched["is_retry"] = True
        return enriched


def _result_to_dict(r: AgentResult) -> Dict[str, Any]:
    return {
        "verdict": r.verdict.value,
        "confidence": r.confidence,
        "latency_ms": round(r.latency_ms, 1),
        "payload": r.payload,
        "traces": [
            {"step": t.step, "source": t.source, "rule_ids": t.rule_ids, "detail": t.detail}
            for t in r.traces
        ],
    }
