"""Chunk re-ranking: resolves conflicts and boosts regulation-backed evidence."""

from __future__ import annotations

from typing import List

from .multi_source_retriever import RetrievedChunk


class ChunkRanker:
    """Re-ranks retrieved chunks to prioritize regulation-backed evidence.

    Ranking strategy:
        1. Chunks with explicit rule_id get a priority boost.
        2. Regulation source > ontology source > case_history source.
        3. Within the same source, rank by relevance_score.
    """

    SOURCE_PRIORITY = {"regulation": 0.15, "ontology": 0.10, "case_history": 0.0}

    def rerank(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        def _score(chunk: RetrievedChunk) -> float:
            base = chunk.relevance_score
            base += self.SOURCE_PRIORITY.get(chunk.source, 0.0)
            if chunk.rule_id:
                base += 0.10
            return base

        return sorted(chunks, key=_score, reverse=True)
