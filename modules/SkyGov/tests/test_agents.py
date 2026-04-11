"""Unit tests for SkyGov agents."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skygov.agents import ComplianceAgent, RiskAssessmentAgent, ExplanationAgent, AuditAgent
from skygov.agents.base_agent import AgentVerdict


class TestComplianceAgent:
    def test_violation_detected(self):
        agent = ComplianceAgent()
        context = {
            "uav_id": "UAV-TEST-001",
            "telemetry": {"wind_resistance": 5, "current_env_wind": 7, "battery": 80},
        }
        result = agent._timed_execute(context)
        assert result.verdict == AgentVerdict.VIOLATION
        assert result.confidence == 1.0
        assert len(result.traces) > 0

    def test_safe_scenario(self):
        agent = ComplianceAgent()
        context = {
            "uav_id": "UAV-TEST-002",
            "telemetry": {"wind_resistance": 6, "current_env_wind": 3, "battery": 90},
        }
        result = agent._timed_execute(context)
        assert result.verdict == AgentVerdict.SAFE

    def test_battery_critical(self):
        agent = ComplianceAgent()
        context = {
            "uav_id": "UAV-TEST-003",
            "telemetry": {"wind_resistance": 6, "current_env_wind": 3, "battery": 10},
        }
        result = agent._timed_execute(context)
        assert result.verdict == AgentVerdict.VIOLATION


class TestRiskAssessmentAgent:
    def test_mock_risk_assessment(self):
        agent = RiskAssessmentAgent()
        context = {
            "uav_id": "UAV-TEST-001",
            "telemetry": {"wind_resistance": 5, "current_env_wind": 4},
            "mission": {"mission_type": "logistics"},
        }
        result = agent._timed_execute(context)
        assert result.verdict in (AgentVerdict.SAFE, AgentVerdict.RISK, AgentVerdict.UNCERTAIN)
        assert "risk_level" in result.payload


class TestExplanationAgent:
    def test_mock_explanation(self):
        agent = ExplanationAgent()
        context = {
            "scenario": {"uav_id": "UAV-TEST-001"},
            "compliance_result": {"verdict": "violation"},
            "risk_result": {"risk_level": "high", "confidence": 0.9},
        }
        result = agent._timed_execute(context)
        assert "explanation" in result.payload
        assert len(result.payload["cited_rules"]) > 0


class TestAuditAgent:
    def test_high_quality_passes(self):
        agent = AuditAgent()
        context = {
            "explanation": (
                "根据REG-WIND-001，当前风速超过抗风等级。"
                "根据REG-SAFETY-012，应立即执行返航程序。"
            ),
            "cited_rules": ["REG-WIND-001", "REG-SAFETY-012"],
            "relevant_rules": {"REG-WIND-001", "REG-SAFETY-012"},
        }
        result = agent._timed_execute(context)
        assert result.payload["rar"] > 0.5
        assert result.payload["passed"] is True

    def test_low_quality_fails(self):
        agent = AuditAgent()
        context = {
            "explanation": "该UAV可能存在风险，建议注意安全。天气不太好。",
            "cited_rules": [],
        }
        result = agent._timed_execute(context)
        assert result.payload["rar"] == 0.0
        assert result.payload["passed"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
