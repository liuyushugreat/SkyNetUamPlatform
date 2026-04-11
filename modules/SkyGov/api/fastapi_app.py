"""FastAPI service for SkyGov multi-agent governance system."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .schemas import GovernanceRequest, GovernanceResponse, DecisionOutput, AgentOutput, TraceOutput

from ..skygov.config import SkyGovConfig
from ..skygov.agents import ComplianceAgent, RiskAssessmentAgent, ExplanationAgent, AuditAgent
from ..skygov.orchestrator import WorkflowEngine, TrustProtocol, TASK_FLIGHT_APPROVAL, TASK_REALTIME_MONITOR
from ..skygov.governance import DecisionTracer, ComplianceReportGenerator

app = FastAPI(
    title="SkyGov API",
    description="LLM-driven multi-agent governance for UAM regulatory compliance",
    version="0.1.0",
)

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"
config = SkyGovConfig.from_yaml(CONFIG_PATH) if CONFIG_PATH.exists() else SkyGovConfig()

agents = {
    "compliance": ComplianceAgent(config=config.agents.compliance),
    "risk_assessment": RiskAssessmentAgent(config=config.agents.risk_assessment),
    "explanation": ExplanationAgent(config=config.agents.explanation),
    "audit": AuditAgent(config=config.agents.audit),
}

trust = TrustProtocol()
engine = WorkflowEngine(agents=agents, trust_protocol=trust, max_retries=config.workflow.max_retries)
tracer = DecisionTracer()
reporter = ComplianceReportGenerator()

TASK_MAP = {
    "flight_approval": TASK_FLIGHT_APPROVAL,
    "realtime_monitor": TASK_REALTIME_MONITOR,
}


@app.post("/governance", response_model=GovernanceResponse)
async def run_governance(req: GovernanceRequest):
    request_id = uuid.uuid4().hex[:12]
    context = {
        "uav_id": req.uav_id,
        "telemetry": req.telemetry.model_dump(),
        "mission": req.mission.model_dump(),
        "scenario": {
            "uav_id": req.uav_id,
            **req.telemetry.model_dump(),
            **req.mission.model_dump(),
        },
    }

    task_graph = TASK_MAP.get(req.task_type, TASK_FLIGHT_APPROVAL)
    result = engine.run(task_graph, context)

    record = tracer.create_record(request_id, req.uav_id, context["scenario"])
    for agent_name, agent_result in result.get("agent_results", {}).items():
        tracer.append_agent_output(record, agent_name, agent_result)
    decision = result["decision"]
    tracer.finalize(record, decision["final_verdict"], decision["action"])
    report_md = reporter.to_markdown(record)

    agent_outputs = {}
    for name, ar in result.get("agent_results", {}).items():
        agent_outputs[name] = AgentOutput(
            verdict=ar.get("verdict", ""),
            confidence=ar.get("confidence", 0),
            latency_ms=ar.get("latency_ms", 0),
            payload=ar.get("payload", {}),
            traces=[TraceOutput(**t) for t in ar.get("traces", [])],
        )

    return GovernanceResponse(
        request_id=request_id,
        decision=DecisionOutput(**decision),
        agent_results=agent_outputs,
        total_latency_ms=result["total_latency_ms"],
        retries=result.get("retries", 0),
        report_markdown=report_md,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "agents": list(agents.keys())}
