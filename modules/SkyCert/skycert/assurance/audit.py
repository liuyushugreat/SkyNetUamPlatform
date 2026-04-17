"""Audit artifact logger.

Each call to :class:`SkyCertPipeline` is expected to produce exactly
one audit record, committed to a JSON-Lines file at the configured
path.  A record captures *enough* information for an offline reviewer
to reconstruct:

* the firing rules (if the symbolic engine is available),
* the conformal prediction set and set size,
* the current martingale value and alert flag,
* the final decision and the reasons that led to it.

Reviewers can then ingest the resulting file to build a certification
argument; the fields are deliberately chosen to support the assurance
objectives listed in Section 3 of the paper.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..utils import safe_json
from .policy import AssuranceDecision


class AuditLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")
        self._closed = False
        self._counter = 0

    def log(
        self,
        decision: AssuranceDecision,
        *,
        sample_id: int | str,
        probs: list[float] | None = None,
        rule_trace: list[dict] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "sample_id": sample_id,
            "index": self._counter,
            "decision": asdict(decision),
            "probs": probs,
            "rule_trace": rule_trace or [],
            "extra": extra or {},
        }
        self._fh.write(json.dumps(safe_json(record), ensure_ascii=False) + "\n")
        self._counter += 1

    def close(self) -> None:
        if not self._closed:
            self._fh.close()
            self._closed = True

    def __enter__(self) -> "AuditLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        self.close()
