"""Unit tests for SkyGov workflow engine and trust protocol."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skygov.agents import ComplianceAgent, RiskAssessmentAgent, ExplanationAgent, AuditAgent
from skygov.orchestrator import WorkflowEngine, TrustProtocol, TASK_FLIGHT_APPROVAL


class TestWorkflowEngine:
    def _make_engine(self):
        agents = {
            "compliance": ComplianceAgent(),
            "risk_assessment": RiskAssessmentAgent(),
            "explanation": ExplanationAgent(),
            "audit": AuditAgent(),
        }
        return WorkflowEngine(agents=agents, trust_protocol=TrustProtocol())

    def test_violation_short_circuits(self):
        engine = self._make_engine()
        scenario = {
            "uav_id": "UAV-WF-001",
            "telemetry": {"wind_resistance": 5, "current_env_wind": 8, "battery": 80},
            "scenario": {"uav_id": "UAV-WF-001"},
        }
        result = engine.run(TASK_FLIGHT_APPROVAL, scenario)
        assert result["decision"]["final_verdict"] == "violation"
        assert result["decision"]["action"] == "reject"

    def test_safe_scenario_approves(self):
        engine = self._make_engine()
        scenario = {
            "uav_id": "UAV-WF-002",
            "telemetry": {"wind_resistance": 7, "current_env_wind": 3, "battery": 90},
            "scenario": {"uav_id": "UAV-WF-002"},
        }
        result = engine.run(TASK_FLIGHT_APPROVAL, scenario)
        assert result["decision"]["final_verdict"] in ("safe", "risk", "uncertain")
        assert result["total_latency_ms"] > 0


class TestTrustProtocol:
    def test_veto_overrides(self):
        from skygov.agents.base_agent import AgentResult, AgentVerdict

        protocol = TrustProtocol()
        results = {
            "compliance": AgentResult(
                agent_name="compliance",
                verdict=AgentVerdict.VIOLATION,
                confidence=1.0,
            ),
            "risk_assessment": AgentResult(
                agent_name="risk_assessment",
                verdict=AgentVerdict.SAFE,
                confidence=0.9,
            ),
        }
        decision = protocol.aggregate(results)
        assert decision["final_verdict"] == "violation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
