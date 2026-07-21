"""Martingale-based sequential evidence for drift and corruption detection.

We follow the test-martingale construction of Vovk, Nouretdinov and
Gammerman (2003) for online exchangeability testing.  At each step we
compute a smoothed conformal ``p``-value against the pooled history
(calibration warm-start plus stream so far)

    p_t = ( |{ i : s_i > s_t }| + U_t * |{ i : s_i = s_t }| ) / n_t,
    U_t ~ Uniform(0, 1),

where the randomised tie-break keeps ``p_t`` exactly uniform under
exchangeability, and update a betting martingale

    M_t = prod_{i=1..t} f(p_i),

where ``f`` is any measurable density on ``[0,1]`` with mean 1.  The
default ``f`` is the *simple jumper* used by Ho and Wechsler (2010):

    f_epsilon(p) = epsilon * p^(epsilon - 1),   epsilon in (0,1).

``M`` is a nonnegative martingale under exchangeability, so by Ville's
inequality ``Pr[ sup_t M_t >= lambda ] <= 1 / lambda``. We raise an
ALERT when ``M_t`` crosses the user-specified threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimpleJumperBetting:
    """Simple-jumper betting function ``f(p) = epsilon * p^(epsilon-1)``."""

    epsilon: float = 0.92

    def __post_init__(self) -> None:
        if not (0.0 < self.epsilon < 1.0):
            raise ValueError("epsilon must lie in (0, 1)")

    def __call__(self, p: float) -> float:
        p = max(p, 1e-6)
        return float(self.epsilon * (p ** (self.epsilon - 1.0)))


@dataclass
class MartingaleMonitor:
    """Streaming test-martingale monitor.

    Parameters
    ----------
    threshold:
        Value at which ``M_t`` triggers a drift/corruption alert. By
        Ville's inequality, thresholding at ``lambda`` gives an
        anytime-valid false-positive bound of at most ``1 / lambda``.
    betting:
        A betting function mapping ``p in (0,1] -> R_+`` with expectation
        1 under uniform ``p``. Defaults to the simple jumper.
    seed:
        Seeds the randomised tie-breaker used inside ``update``.  Keeping
        the tie-breaker seeded is important for bit-for-bit reproducible
        experiments.
    """

    threshold: float = 20.0
    betting: SimpleJumperBetting = field(default_factory=SimpleJumperBetting)
    seed: int = 20260417

    # Running state (populated as observations arrive).
    score_history: list[float] = field(default_factory=list)
    martingale: float = 1.0
    trace: list[float] = field(default_factory=list)
    alert_history: list[bool] = field(default_factory=list)
    _max_martingale: float = 1.0
    _rng: np.random.Generator | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def reset(self) -> None:
        self.score_history.clear()
        self.trace.clear()
        self.alert_history.clear()
        self.martingale = 1.0
        self._max_martingale = 1.0
        self._rng = np.random.default_rng(self.seed)

    def warm_start(self, reference: np.ndarray) -> None:
        """Seed the monitor with calibration scores.

        The reference scores are added to ``score_history`` but do *not*
        contribute to the martingale. Subsequent test scores receive
        conformal p-values computed against the pooled history, which
        is the standard *conformal test-martingale* construction.
        """
        self.score_history = list(map(float, reference.tolist()))

    def update(self, score: float) -> dict:
        """Push one nonconformity score and return the current state."""
        t_pre = len(self.score_history)
        self.score_history.append(float(score))
        arr = np.asarray(self.score_history)
        n = arr.size
        rank_strict = float(np.sum(arr > score))
        rank_eq = float(np.sum(arr == score))
        assert self._rng is not None
        u = float(self._rng.uniform(0.0, 1.0))
        p = (rank_strict + u * rank_eq) / n
        p = float(max(min(p, 1.0), 1e-6))
        f_p = self.betting(p)
        self.martingale *= f_p
        self._max_martingale = max(self._max_martingale, self.martingale)
        self.trace.append(self.martingale)
        alert = self.martingale >= self.threshold
        self.alert_history.append(alert)
        return {
            "p_value": p,
            "f_p": f_p,
            "martingale": self.martingale,
            "alert": alert,
            "t": t_pre + 1,
        }

    def cumulative_alerts(self) -> int:
        return int(sum(self.alert_history))

    def max_value(self) -> float:
        return float(self._max_martingale)

    def detection_delay(self, change_point: int) -> int | None:
        """Return the number of steps after ``change_point`` until the
        first alert, or ``None`` if no alert was ever raised."""
        for i, alert in enumerate(self.alert_history):
            if i >= change_point and alert:
                return i - change_point
        return None
