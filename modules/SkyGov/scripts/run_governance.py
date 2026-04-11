"""Single-scenario governance demo: runs the full 4-agent workflow on a test case."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skygov.config import SkyGovConfig
from skygov.agents import ComplianceAgent, RiskAssessmentAgent, ExplanationAgent, AuditAgent
from skygov.orchestrator import WorkflowEngine, TrustProtocol, TASK_FLIGHT_APPROVAL
from skygov.governance import DecisionTracer, ComplianceReportGenerator


def main():
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    cfg = SkyGovConfig.from_yaml(config_path) if config_path.exists() else SkyGovConfig()

    agents = {
        "compliance": ComplianceAgent(config=cfg.agents.compliance),
        "risk_assessment": RiskAssessmentAgent(config=cfg.agents.risk_assessment),
        "explanation": ExplanationAgent(config=cfg.agents.explanation),
        "audit": AuditAgent(config=cfg.agents.audit),
    }

    trust = TrustProtocol()
    engine = WorkflowEngine(agents=agents, trust_protocol=trust)
    tracer = DecisionTracer()
    reporter = ComplianceReportGenerator()

    # ── Test scenario: UAV in high wind ──
    scenario = {
        "uav_id": "UAV-DEMO-001",
        "telemetry": {
            "wind_resistance": 5,
            "current_env_wind": 7,
            "battery": 65,
        },
        "mission": {
            "mission_type": "logistics",
            "payload_kg": 3.5,
            "destination": "SECTOR-B12",
        },
        "scenario": {
            "uav_id": "UAV-DEMO-001",
            "wind_resistance": 5,
            "current_env_wind": 7,
            "battery": 65,
        },
    }

    print("=" * 60)
    print("  SkyGov Multi-Agent Governance Demo")
    print("=" * 60)
    print(f"\nScenario: {json.dumps(scenario['telemetry'], ensure_ascii=False)}")
    print()

    result = engine.run(TASK_FLIGHT_APPROVAL, scenario)

    record = tracer.create_record("demo-001", "UAV-DEMO-001", scenario["scenario"])
    for name, ar in result.get("agent_results", {}).items():
        tracer.append_agent_output(record, name, ar)
    decision = result["decision"]
    tracer.finalize(record, decision["final_verdict"], decision["action"])

    print("\n>>> Decision:")
    print(json.dumps(decision, ensure_ascii=False, indent=2))

    print(f"\n>>> Total latency: {result['total_latency_ms']:.1f}ms")

    report = reporter.to_markdown(record)
    print("\n>>> Compliance Report:")
    print(report)

    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "demo_decision.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "demo_report.md").write_text(report, encoding="utf-8")
    print(f"\n>>> Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
