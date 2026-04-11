"""Context window builder: assembles retrieved chunks into an LLM-ready prompt segment."""

from __future__ import annotations

from typing import List

from .multi_source_retriever import RetrievedChunk


class ContextBuilder:
    """Builds the evidence section of LLM prompts under a token budget.

    Ensures the total evidence text fits within the configured context budget,
    truncating lower-priority chunks when necessary.
    """

    def __init__(self, budget_tokens: int = 3072, chars_per_token: float = 1.5):
        self.budget_tokens = budget_tokens
        self.chars_per_token = chars_per_token

    @property
    def budget_chars(self) -> int:
        return int(self.budget_tokens * self.chars_per_token)

    def build(self, chunks: List[RetrievedChunk]) -> str:
        lines: List[str] = []
        used = 0
        for i, chunk in enumerate(chunks):
            prefix = f"[证据{i + 1}]"
            if chunk.rule_id:
                prefix += f" ({chunk.rule_id})"
            prefix += f" [{chunk.source}]"
            line = f"{prefix} {chunk.text}"

            if used + len(line) > self.budget_chars:
                break
            lines.append(line)
            used += len(line)

        return "\n".join(lines)
