"""Multi-source retriever: fetches evidence from ontology, regulations, and case history."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rdflib import Graph, Namespace

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    source: str  # "ontology" | "regulation" | "case_history"
    rule_id: Optional[str] = None
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiSourceRetriever:
    """Retrieves evidence chunks from multiple knowledge sources.

    Sources:
        1. Ontology (RDF/SPARQL) — structured UAM domain knowledge
        2. Regulation store — civil aviation rules and compliance clauses
        3. Case history — past governance decisions and incident records
    """

    def __init__(
        self,
        ontology_graph: Optional[Graph] = None,
        regulation_store: Optional[Any] = None,
        case_store: Optional[Any] = None,
        top_k: int = 5,
    ):
        self.ontology_graph = ontology_graph
        self.regulation_store = regulation_store
        self.case_store = case_store
        self.top_k = top_k
        self.ns = "http://github.com/liuyushugreat/SkyNetUamPlatform/ontology#"

    def retrieve(self, context: Dict[str, Any]) -> List[RetrievedChunk]:
        """Retrieve top-k evidence chunks across all sources."""
        chunks: List[RetrievedChunk] = []
        chunks.extend(self._retrieve_from_ontology(context))
        chunks.extend(self._retrieve_from_regulations(context))
        chunks.extend(self._retrieve_from_cases(context))
        chunks.sort(key=lambda c: c.relevance_score, reverse=True)
        return chunks[: self.top_k]

    def _retrieve_from_ontology(self, context: Dict[str, Any]) -> List[RetrievedChunk]:
        """SPARQL-based retrieval from the UAM knowledge graph."""
        if self.ontology_graph is None:
            return self._mock_ontology_chunks(context)

        uav_id = context.get("uav_id", "")
        chunks = []
        query = f"""
            PREFIX skynet: <{self.ns}>
            SELECT ?prop ?val
            WHERE {{
                skynet:{uav_id} ?prop ?val .
            }}
            LIMIT 10
        """
        try:
            for row in self.ontology_graph.query(query):
                prop_name = str(row.prop).split("#")[-1]
                chunks.append(
                    RetrievedChunk(
                        text=f"{uav_id}.{prop_name} = {row.val}",
                        source="ontology",
                        relevance_score=0.9,
                    )
                )
        except Exception as e:
            logger.warning("Ontology retrieval failed: %s", e)

        return chunks

    def _retrieve_from_regulations(self, context: Dict[str, Any]) -> List[RetrievedChunk]:
        """Retrieve relevant regulation clauses."""
        if self.regulation_store is not None:
            # TODO: integrate with actual regulation vector store
            pass
        return self._mock_regulation_chunks(context)

    def _retrieve_from_cases(self, context: Dict[str, Any]) -> List[RetrievedChunk]:
        """Retrieve similar historical governance cases."""
        if self.case_store is not None:
            # TODO: integrate with case history database
            pass
        return self._mock_case_chunks(context)

    def _mock_ontology_chunks(self, context: Dict[str, Any]) -> List[RetrievedChunk]:
        uav_id = context.get("uav_id", "UAV-001")
        telemetry = context.get("telemetry", {})
        return [
            RetrievedChunk(
                text=f"UAV {uav_id} 型号最大抗风等级: {telemetry.get('wind_resistance', 5)}级",
                source="ontology",
                rule_id="REG-WIND-001",
                relevance_score=0.95,
            ),
            RetrievedChunk(
                text=f"UAV {uav_id} 当前电量: {telemetry.get('battery', 80)}%",
                source="ontology",
                relevance_score=0.80,
            ),
        ]

    def _mock_regulation_chunks(self, context: Dict[str, Any]) -> List[RetrievedChunk]:
        return [
            RetrievedChunk(
                text="《低空飞行安全管理规定》第12条：环境风速超过机型抗风等级时，应立即执行返航或迫降程序。",
                source="regulation",
                rule_id="REG-SAFETY-012",
                relevance_score=0.92,
            ),
            RetrievedChunk(
                text="《无人机运行管理暂行条例》第8条：电池电量低于15%时，禁止继续执行非紧急任务。",
                source="regulation",
                rule_id="REG-BAT-001",
                relevance_score=0.85,
            ),
        ]

    def _mock_case_chunks(self, context: Dict[str, Any]) -> List[RetrievedChunk]:
        return [
            RetrievedChunk(
                text="案例2025-A317：某物流UAV在6级风中失稳坠落，事后调查确认未遵循REG-WIND-001。",
                source="case_history",
                relevance_score=0.70,
            ),
        ]
