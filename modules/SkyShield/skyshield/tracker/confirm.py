"""M-of-N detection confirmation for radar tracks.

Each time the fusion plane accepts a valid packet for a track id, the
confirmer records a detection timestamp.  A track is *confirmed* once
``m`` out of the last ``n`` revisit slots contain a detection.  Once
confirmed, a track remains confirmed until explicitly dropped, which
gives the decision plane a stable contract to reason about deadlines.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Tuple


@dataclass
class ConfirmationState:
    hits: Deque[float]       # timestamps of accepted detections (ms)
    confirmed: bool = False
    confirmed_at_ms: float = 0.0


class MofNConfirmer:
    def __init__(self, m_of_n: Tuple[int, int]):
        self.m, self.n = m_of_n
        self._states: Dict[int, ConfirmationState] = {}

    def observe(self, track_id: int, now_ms: float) -> ConfirmationState:
        st = self._states.get(track_id)
        if st is None:
            st = ConfirmationState(hits=deque(maxlen=self.n))
            self._states[track_id] = st
        st.hits.append(now_ms)
        if not st.confirmed and len(st.hits) >= self.m:
            # Window of length n carries >= m hits implicitly by deque length.
            st.confirmed = True
            st.confirmed_at_ms = now_ms
        return st

    def is_confirmed(self, track_id: int) -> bool:
        st = self._states.get(track_id)
        return bool(st and st.confirmed)

    def confirmed_at(self, track_id: int) -> float:
        st = self._states.get(track_id)
        return st.confirmed_at_ms if st and st.confirmed else float("inf")

    def drop(self, track_id: int) -> None:
        self._states.pop(track_id, None)
