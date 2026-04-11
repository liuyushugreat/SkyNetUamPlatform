"""Pydantic request/response schemas for the SkyGov FastAPI service."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TelemetryInput(BaseModel):
    wind_resistance: int = Field(5, description="UAV max wind resistance level")
    current_env_wind: int = Field(3, description="Current environment wind level")
    battery: int = Field(80, description="Battery percentage")


class MissionInput(BaseModel):
    mission_type: str = Field("logistics", description="Mission type")
    payload_kg: float = Field(2.0, description="Payload weight in kg")
    destination: str = Field("", description="Destination identifier")


class GovernanceRequest(BaseModel):
    uav_id: str = Field(..., description="UAV identifier")
    telemetry: TelemetryInput = Field(default_factory=TelemetryInput)
    mission: MissionInput = Field(default_factory=MissionInput)
    task_type: str = Field("flight_approval", description="Governance task type")


class TraceOutput(BaseModel):
    step: str
    source: str
    rule_ids: List[str] = []
    detail: str = ""


class AgentOutput(BaseModel):
    verdict: str
    confidence: float
    latency_ms: float
    payload: Dict[str, Any] = {}
    traces: List[TraceOutput] = []


class DecisionOutput(BaseModel):
    final_verdict: str
    reason: str
    confidence: float
    quality_check: str
    action: str


class GovernanceResponse(BaseModel):
    request_id: str
    decision: DecisionOutput
    agent_results: Dict[str, AgentOutput] = {}
    total_latency_ms: float
    retries: int = 0
    report_markdown: Optional[str] = None
