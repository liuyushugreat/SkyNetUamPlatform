"""Abstention / alert / escalation decision policy.

The policy consumes, per operation request:

* the conformal prediction set,
* the current martingale state (max value and alert flag),
* a ``top1`` neural-symbolic proposal,

and returns an ``AssuranceDecision`` that the platform must respect:

* ``ACCEPT``    — emit the top-1 risk label with the accompanying set,
* ``ABSTAIN``   — do not commit to a single label (return the set),
* ``ALERT``     — same as ABSTAIN plus operator notification,
* ``ESCALATE``  — hand the decision to a higher authority / human.

Rules:

* if the set is larger than ``max_set_fraction * K`` classes -> ABSTAIN,
* if the martingale is above threshold -> ALERT (ABSTAIN semantics),
* if both conditions hold and ``escalate_on_martingale`` is set, ESCALATE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class DecisionKind(str, Enum):
    ACCEPT = "ACCEPT"
    ABSTAIN = "ABSTAIN"
    ALERT = "ALERT"
    ESCALATE = "ESCALATE"


@dataclass
class AssuranceDecision:
    kind: DecisionKind
    top1: int
    prediction_set: list[int]
    set_size: int
    martingale: float
    martingale_alert: bool
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AssurancePolicy:
    num_classes: int
    max_set_fraction: float = 0.75
    escalate_on_martingale: bool = True

    def __post_init__(self) -> None:
        # Values slightly above 1.0 are allowed so that ablation studies can
        # effectively disable the set-size branch without touching the code.
        if not (0.0 < self.max_set_fraction <= 2.0):
            raise ValueError("max_set_fraction must lie in (0, 2]")

    def decide(
        self,
        probs_row: np.ndarray,
        set_mask: np.ndarray,
        martingale_value: float,
        martingale_alert: bool,
    ) -> AssuranceDecision:
        top1 = int(np.argmax(probs_row))
        set_indices = [int(i) for i, on in enumerate(set_mask) if on]
        set_size = len(set_indices)

        reasons: list[str] = []
        set_fraction = set_size / max(self.num_classes, 1)
        is_ambiguous = set_fraction >= self.max_set_fraction
        if is_ambiguous:
            reasons.append(
                f"set_size_fraction={set_fraction:.2f} "
                f">= max_set_fraction={self.max_set_fraction:.2f}"
            )
        if martingale_alert:
            reasons.append(
                f"martingale={martingale_value:.2f} exceeds threshold"
            )

        if is_ambiguous and martingale_alert and self.escalate_on_martingale:
            kind = DecisionKind.ESCALATE
        elif martingale_alert:
            kind = DecisionKind.ALERT
        elif is_ambiguous:
            kind = DecisionKind.ABSTAIN
        else:
            kind = DecisionKind.ACCEPT

        return AssuranceDecision(
            kind=kind,
            top1=top1,
            prediction_set=set_indices,
            set_size=set_size,
            martingale=float(martingale_value),
            martingale_alert=bool(martingale_alert),
            reasons=reasons,
        )
