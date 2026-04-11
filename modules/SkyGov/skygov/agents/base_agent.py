"""Abstract base class for all SkyGov agents."""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AgentVerdict(str, Enum):
    SAFE = "safe"
    RISK = "risk"
    VIOLATION = "violation"
    UNCERTAIN = "uncertain"
    VETO = "veto"


@dataclass
class TraceEntry:
    """Single step in the decision trace chain."""

    step: str
    source: str
    rule_ids: List[str] = field(default_factory=list)
    detail: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentResult:
    """Standardized output from any SkyGov agent."""

    agent_name: str
    verdict: AgentVerdict
    confidence: float = 0.0
    payload: Dict[str, Any] = field(default_factory=dict)
    traces: List[TraceEntry] = field(default_factory=list)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    latency_ms: float = 0.0

    @property
    def is_blocking(self) -> bool:
        return self.verdict in (AgentVerdict.VIOLATION, AgentVerdict.VETO)


class BaseAgent(ABC):
    """All SkyGov agents inherit from this class."""

    name: str = "base"

    def __init__(self, config: Any = None):
        self.config = config
        self.logger = logging.getLogger(f"skygov.agent.{self.name}")

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Run the agent on the given governance context.

        Args:
            context: Flight scenario data including UAV info, airspace,
                     weather, regulations, and any prior agent outputs.
        Returns:
            Standardized AgentResult.
        """

    def _timed_execute(self, context: Dict[str, Any]) -> AgentResult:
        t0 = time.perf_counter()
        result = self.execute(context)
        result.latency_ms = (time.perf_counter() - t0) * 1000
        self.logger.info(
            "%s finished: verdict=%s confidence=%.2f latency=%.1fms",
            self.name,
            result.verdict.value,
            result.confidence,
            result.latency_ms,
        )
        return result
