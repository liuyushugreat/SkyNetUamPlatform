"""Governance engine — assigns rights profiles to asset units.

Policy rules (transparent, not black-box)
-----------------------------------------
* Flights with **violations** -> ``NON_TRANSFERABLE``, desensitization required.
* Flights with **anomalies** or incomplete missions -> ``INTERNAL_ONLY``.
* Clean, completed flights with low risk -> eligible for
  ``TRADABLE_AFTER_DESENSITIZATION``; asset class is upgraded based on
  mission-type keyword matching.

Revenue-split ratios come from ``GovernanceEngine``'s constructor or are
overridden per-call — they are **never hard-coded** inside business logic.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Dict, List, Optional

from ..models.asset_unit import FlightAssetUnit
from ..models.enums import (
    AssetClass,
    AssetStatus,
    DataCategory,
    UsageLevel,
)
from ..models.rights import RetentionPolicy, RevenueParticipant, RightsProfile


class GovernanceEngine:
    """Assigns a :class:`RightsProfile` and optionally upgrades the asset class.

    Raises
    ------
    ValueError
        If *unit* has no attached evidence when ``govern()`` is called.
    """

    def __init__(
        self,
        default_owner: str = "platform",
        default_retention_days: int = 365 * 3,
        default_revenue_split: Optional[List[Dict[str, object]]] = None,
    ):
        self.default_owner = default_owner
        self.default_retention_days = default_retention_days
        self._default_split = default_revenue_split or [
            {"party_id": "platform", "role": "platform", "share_pct": 30.0},
            {"party_id": "operator", "role": "operator", "share_pct": 50.0},
            {"party_id": "data_processor", "role": "data_processor", "share_pct": 20.0},
        ]

    def govern(
        self,
        unit: FlightAssetUnit,
        owner: Optional[str] = None,
        operator_id: str = "",
    ) -> FlightAssetUnit:
        """Apply governance rules and attach a :class:`RightsProfile`."""
        if unit.evidence is None:
            raise ValueError(
                f"Cannot govern asset unit {unit.asset_unit_id}: "
                "evidence package is missing (run EvidenceBuilder first)"
            )

        evidence = unit.evidence
        has_violations = bool(evidence.mission_result.violations)
        has_anomalies = bool(evidence.mission_result.anomalies)
        mission_ok = bool(evidence.mission_result.completed)

        # --- determine usage level (rule is transparent) ---
        if has_violations:
            permitted = [UsageLevel.NON_TRANSFERABLE]
            tradable = False
            desens = True
        elif has_anomalies or not mission_ok:
            permitted = [UsageLevel.INTERNAL_ONLY]
            tradable = False
            desens = False
        else:
            permitted = [
                UsageLevel.INTERNAL_ONLY,
                UsageLevel.TRADABLE_AFTER_DESENSITIZATION,
            ]
            tradable = True
            desens = True

        # --- data categories ---
        categories = [DataCategory.RAW_TELEMETRY]
        if mission_ok and not has_violations:
            categories.append(DataCategory.DERIVED_FEATURES)
        if tradable:
            categories.append(DataCategory.TRAINING_SAMPLE)

        # --- revenue participants (read from config, not hard-coded) ---
        real_owner = owner or self.default_owner
        participants = [
            RevenueParticipant(**entry)  # type: ignore[arg-type]
            for entry in self._default_split
        ]
        if operator_id:
            for p in participants:
                if p.role == "operator":
                    p.party_id = operator_id

        profile = RightsProfile(
            owner=real_owner,
            controller=real_owner,
            contributors=[operator_id] if operator_id else [],
            data_categories=categories,
            permitted_uses=permitted,
            desensitization_required=desens,
            tradable=tradable,
            retention=RetentionPolicy(
                max_retention_days=self.default_retention_days,
            ),
            revenue_split=participants,
        )

        # --- optionally upgrade asset class ---
        asset_class = unit.asset_class
        if tradable and mission_ok and unit.risk_score < 0.3:
            asset_class = self._suggest_asset_class(unit)

        unit.rights_profile = profile
        unit.asset_class = asset_class
        unit.compliance_score = self._compliance_score(unit)
        unit.status = AssetStatus.GOVERNED
        unit.updated_at = datetime.now(UTC)
        return unit

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _suggest_asset_class(unit: FlightAssetUnit) -> AssetClass:
        mt = (unit.mission_type or "").lower()
        if "inspection" in mt or "maintenance" in mt:
            return AssetClass.MAINTENANCE_SAMPLE
        if "survey" in mt or "route" in mt:
            return AssetClass.ROUTE_OPTIMIZATION_SAMPLE
        if "weather" in mt:
            return AssetClass.WEATHER_OPERATION_SAMPLE
        if "compliance" in mt or "audit" in mt:
            return AssetClass.COMPLIANCE_RECORD
        return AssetClass.FLIGHT_EVIDENCE

    @staticmethod
    def _compliance_score(unit: FlightAssetUnit) -> float:
        """Transparent compliance heuristic.

        Starts at 1.0 and deducts:
        - 0.20 per violation
        - 0.05 per anomaly
        - 0.10 if mission was not completed
        """
        if not unit.evidence:
            return 0.0
        mr = unit.evidence.mission_result
        score = 1.0
        score -= len(mr.violations) * 0.2
        score -= len(mr.anomalies) * 0.05
        if not mr.completed:
            score -= 0.1
        return max(score, 0.0)
