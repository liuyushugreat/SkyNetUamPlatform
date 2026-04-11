"""Centralized configuration for SkyGov multi-agent governance system."""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class LLMConfig:
    provider: str = "deepseek"
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout_seconds: int = 30


@dataclass
class RAGConfig:
    top_k: int = 5
    chunk_max_tokens: int = 512
    context_budget_tokens: int = 3072
    rerank_enabled: bool = True
    sources: List[str] = field(
        default_factory=lambda: ["ontology", "regulation", "case_history"]
    )


@dataclass
class ComplianceAgentConfig:
    enabled: bool = True
    veto_power: bool = True
    sparql_timeout_ms: int = 500


@dataclass
class RiskAssessmentAgentConfig:
    enabled: bool = True
    rag_top_k: int = 5
    confidence_threshold: float = 0.7


@dataclass
class ExplanationAgentConfig:
    enabled: bool = True
    max_explanation_tokens: int = 512
    require_rule_citation: bool = True


@dataclass
class AuditAgentConfig:
    enabled: bool = True
    rar_threshold: float = 0.8
    lec_threshold: float = 0.6
    ucr_threshold: float = 0.1
    re_retrieval_on_failure: bool = True


@dataclass
class AgentsConfig:
    compliance: ComplianceAgentConfig = field(default_factory=ComplianceAgentConfig)
    risk_assessment: RiskAssessmentAgentConfig = field(
        default_factory=RiskAssessmentAgentConfig
    )
    explanation: ExplanationAgentConfig = field(default_factory=ExplanationAgentConfig)
    audit: AuditAgentConfig = field(default_factory=AuditAgentConfig)


@dataclass
class WorkflowConfig:
    default_task: str = "flight_approval"
    timeout_seconds: int = 60
    max_retries: int = 2
    human_escalation_enabled: bool = True


@dataclass
class GovernanceConfig:
    trace_all_decisions: bool = True
    report_format: str = "json"
    hallucination_check_enabled: bool = True


@dataclass
class OntologyConfig:
    path: str = "../SkyKg/SkyNet_Knowledge_Engine/ontology/skynet_core.ttl"
    namespace: str = (
        "http://github.com/liuyushugreat/SkyNetUamPlatform/ontology#"
    )


@dataclass
class SkyGovConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    agents: AgentsConfig = field(default_factory=AgentsConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    governance: GovernanceConfig = field(default_factory=GovernanceConfig)
    ontology: OntologyConfig = field(default_factory=OntologyConfig)
    output_dir: str = "outputs"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SkyGovConfig":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        cfg = cls()
        _recursive_update(cfg, raw)
        return cfg

    def to_yaml(self, path: str | Path) -> None:
        from dataclasses import asdict

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)


def _recursive_update(obj, mapping: dict):
    """Recursively set dataclass fields from a nested dict."""
    if mapping is None:
        return
    for key, value in mapping.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if isinstance(value, dict) and hasattr(current, "__dataclass_fields__"):
            _recursive_update(current, value)
        else:
            setattr(obj, key, value)
