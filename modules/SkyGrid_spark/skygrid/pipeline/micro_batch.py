"""Micro-batch former with a timeout (§5.3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class MicroBatch:
    batch_id: int
    op_name: str
    site: str
    events: list                          # list of Event
    created_at_ms: float
    flushed_at_ms: float | None = None

    @property
    def size(self) -> int:
        return len(self.events)


@dataclass
class MicroBatcher:
    """Per-(op, site) micro-batcher.

    Calling :py:meth:`add` appends one event.  A batch is flushed when
    either (a) it has reached ``max_size`` or (b) more than
    ``timeout_ms`` has elapsed since it was created.
    """

    op_name: str
    site: str
    max_size: int = 64
    timeout_ms: float = 8.0
    _open: MicroBatch | None = None
    _next_id: int = 0
    _pending: list[MicroBatch] = field(default_factory=list)

    def _new_batch(self, now_ms: float) -> MicroBatch:
        self._open = MicroBatch(
            batch_id=self._next_id,
            op_name=self.op_name,
            site=self.site,
            events=[],
            created_at_ms=now_ms,
        )
        self._next_id += 1
        return self._open

    def add(self, event, now_ms: float) -> MicroBatch | None:
        if self._open is None:
            self._new_batch(now_ms)
        assert self._open is not None
        self._open.events.append(event)
        if len(self._open.events) >= self.max_size:
            return self.flush(now_ms)
        return None

    def tick(self, now_ms: float) -> MicroBatch | None:
        if (self._open is not None and
                (now_ms - self._open.created_at_ms) >= self.timeout_ms and
                len(self._open.events) > 0):
            return self.flush(now_ms)
        return None

    def flush(self, now_ms: float) -> MicroBatch | None:
        if self._open is None or not self._open.events:
            return None
        self._open.flushed_at_ms = now_ms
        closed = self._open
        self._open = None
        self._pending.append(closed)
        return closed

    def drain(self) -> Iterator[MicroBatch]:
        for b in self._pending:
            yield b
        self._pending.clear()
