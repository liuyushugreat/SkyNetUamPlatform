"""Revenue splitting logic.

Given a gross revenue amount and a :class:`SettlementRule`, computes
per-participant amounts and returns a list of :class:`SplitEntry`.
"""

from __future__ import annotations

from typing import List

from ..models.settlement import SettlementRule, SplitEntry


class RevenueSplitter:
    """Stateless helper that divides gross revenue among participants."""

    def split(
        self,
        gross_amount: float,
        rule: SettlementRule,
    ) -> List[SplitEntry]:
        """
        Returns a new list of :class:`SplitEntry` with ``amount`` filled in.

        The split normalises ``share_pct`` so that the total always equals 100 %
        even if the input values don't sum to exactly 100.
        """
        if not rule.participants:
            return []

        total_pct = sum(p.share_pct for p in rule.participants) or 1.0
        results: List[SplitEntry] = []
        for p in rule.participants:
            norm_pct = p.share_pct / total_pct * 100
            amount = round(gross_amount * norm_pct / 100.0, 6)
            results.append(
                SplitEntry(
                    party_id=p.party_id,
                    role=p.role,
                    share_pct=round(norm_pct, 4),
                    amount=amount,
                )
            )
        return results
