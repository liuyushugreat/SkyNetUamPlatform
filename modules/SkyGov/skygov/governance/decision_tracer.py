"""Decision chain traceability: every reasoning step linked to regulation clause IDs."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class DecisionRecord:
    """Immutable record of a single governance decision."""

    request_id: str
    uav_id: str
    timestamp: float = field(default_factory=time.time)
    scenario: Dict[str, Any] = field(default_factory=dict)
    agent_chain: List[Dict[str, Any]] = field(default_factory=list)
    final_verdict: str = ""
    final_action: str = ""
    cited_rules: List[str] = field(default_factory=list)
    explanation: str = ""
    quality_scores: Dict[str, float] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, default=str)


class DecisionTracer:
    """Accumulates agent outputs into an auditable decision record."""

    def __init__(self):
        self._records: List[DecisionRecord] = []

    def create_record(
        self, request_id: str, uav_id: str, scenario: Dict[str, Any]
    ) -> DecisionRecord:
        record = DecisionRecord(
            request_id=request_id, uav_id=uav_id, scenario=scenario
        )
        self._records.append(record)
        return record

    def append_agent_output(
        self, record: DecisionRecord, agent_name: str, result_dict: Dict[str, Any]
    ):
        record.agent_chain.append(
            {"agent": agent_name, "timestamp": time.time(), **result_dict}
        )

    def finalize(
        self,
        record: DecisionRecord,
        verdict: str,
        action: str,
        explanation: str = "",
        cited_rules: Optional[List[str]] = None,
        quality_scores: Optional[Dict[str, float]] = None,
    ):
        record.final_verdict = verdict
        record.final_action = action
        record.explanation = explanation
        record.cited_rules = cited_rules or []
        record.quality_scores = quality_scores or {}

    @property
    def records(self) -> List[DecisionRecord]:
        return list(self._records)

    def export_all(self) -> str:
        return json.dumps(
            [asdict(r) for r in self._records],
            ensure_ascii=False,
            indent=2,
            default=str,
        )
