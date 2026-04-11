"""Unit tests for governance layer: tracing, hallucination guard, reports."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skygov.governance import DecisionTracer, HallucinationGuard, ComplianceReportGenerator


class TestDecisionTracer:
    def test_create_and_finalize(self):
        tracer = DecisionTracer()
        record = tracer.create_record("req-001", "UAV-001", {"wind": 7})
        tracer.append_agent_output(record, "compliance", {"verdict": "violation"})
        tracer.finalize(record, "violation", "reject", cited_rules=["REG-WIND-001"])

        assert record.final_verdict == "violation"
        assert len(record.agent_chain) == 1
        assert "REG-WIND-001" in record.cited_rules

    def test_export_json(self):
        tracer = DecisionTracer()
        record = tracer.create_record("req-002", "UAV-002", {})
        tracer.finalize(record, "safe", "approve")
        exported = tracer.export_all()
        assert "req-002" in exported


class TestHallucinationGuard:
    def test_valid_rules_pass(self):
        guard = HallucinationGuard()
        result = guard.check("根据REG-WIND-001，风速超限。REG-SAFETY-012要求返航。")
        assert result["passed"] is True
        assert result["hallucination_rate"] == 0.0

    def test_invalid_rule_detected(self):
        guard = HallucinationGuard()
        result = guard.check("根据REG-FAKE-999，该UAV应停飞。")
        assert result["passed"] is False
        assert "REG-FAKE-999" in result["invalid_rules"]


class TestComplianceReport:
    def test_markdown_generation(self):
        tracer = DecisionTracer()
        record = tracer.create_record("req-003", "UAV-003", {"wind": 5})
        tracer.finalize(
            record, "safe", "approve",
            explanation="场景安全，无违规。",
            quality_scores={"rar": 1.0, "lec": 1.0, "ucr": 0.0},
        )
        reporter = ComplianceReportGenerator()
        md = reporter.to_markdown(record)
        assert "UAV-003" in md
        assert "SkyGov" in md


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
