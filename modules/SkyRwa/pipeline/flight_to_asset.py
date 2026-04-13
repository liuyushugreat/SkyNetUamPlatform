"""End-to-end flight-to-asset pipeline orchestrator.

Connects every layer of the SkyRwa module into a single, sequential pipeline::

    FlightIngestRecord
      1. ingest       -> FlightAssetUnit   (INGESTED)
      2. provenance   -> FlightEvidencePackage attached (EVIDENCE_BUILT)
      3. governance   -> RightsProfile attached        (GOVERNED)
      4. valuation    -> ValuationResultV2 attached     (VALUATED)
      5. settlement   -> SettlementRule attached        (ready for revenue)

Each step is independently replaceable via constructor injection.

The pipeline is designed to run **asynchronously / post-hoc** — it must
never block real-time flight control loops.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..ingest.flight_ingestor import FlightIngestRecord, FlightIngestor
from ..models.asset_unit import FlightAssetUnit
from ..models.settlement import SettlementRule, SplitEntry
from ..provenance.evidence_builder import EvidenceBuilder
from ..rights.governance import GovernanceEngine
from ..valuation.base import AbstractAssetValuationEngine
from ..valuation.rule_engine import RuleBasedValuationEngine

logger = logging.getLogger(__name__)


class FlightToAssetPipeline:
    """Orchestrates the full ingest -> valuation pipeline.

    All components can be injected; defaults are provided for out-of-the-box
    usage.

    Raises
    ------
    ValueError
        If any pipeline step encounters invalid data (propagated from the
        underlying engines).
    """

    def __init__(
        self,
        ingestor: Optional[FlightIngestor] = None,
        evidence_builder: Optional[EvidenceBuilder] = None,
        governance_engine: Optional[GovernanceEngine] = None,
        valuation_engine: Optional[AbstractAssetValuationEngine] = None,
        default_settlement_rule: Optional[SettlementRule] = None,
        signer_id: Optional[str] = None,
    ):
        self.ingestor = ingestor or FlightIngestor()
        self.evidence_builder = evidence_builder or EvidenceBuilder()
        self.governance = governance_engine or GovernanceEngine()
        self.valuation = valuation_engine or RuleBasedValuationEngine()
        self.default_settlement_rule = default_settlement_rule
        self.signer_id = signer_id

    def run(
        self,
        record: FlightIngestRecord,
        *,
        owner: Optional[str] = None,
        settlement_rule: Optional[SettlementRule] = None,
    ) -> FlightAssetUnit:
        """Execute the full pipeline and return a fully-populated asset unit.

        Steps
        -----
        1. **Ingest** — normalise raw flight data into a FlightAssetUnit.
        2. **Provenance** — build FlightEvidencePackage with SHA-256 digest.
        3. **Governance** — assign RightsProfile and compliance score.
        4. **Valuation** — compute DataQualityScore + AssetValueScore.
        5. **Settlement rule** — attach revenue-split configuration.
        """
        if not record.flight_id:
            raise ValueError("FlightIngestRecord.flight_id must not be empty")

        # 1. Ingest
        unit = self.ingestor.ingest(record)
        logger.info(
            "[1/5] Ingested flight_id=%s -> asset_unit_id=%s  status=%s",
            record.flight_id, unit.asset_unit_id, unit.status.value,
        )

        # 2. Provenance
        self.evidence_builder.build(unit, record, signer_id=self.signer_id)
        digest_short = (unit.evidence.digest_hash[:16] + "...") if unit.evidence else "?"
        logger.info("[2/5] Evidence built  digest=%s", digest_short)

        # 3. Governance
        operator_id = record.operator_id or ""
        self.governance.govern(unit, owner=owner, operator_id=operator_id)
        logger.info(
            "[3/5] Governed  class=%s  tradable=%s  compliance=%.2f",
            unit.asset_class.value,
            unit.rights_profile.tradable if unit.rights_profile else "?",
            unit.compliance_score,
        )

        # 4. Valuation
        result = self.valuation.evaluate(unit)
        logger.info(
            "[4/5] Valuated  value=%.4f %s  quality=%.4f  confidence=%.4f",
            result.estimated_value,
            result.currency,
            result.quality_score.overall,
            result.confidence,
        )

        # 5. Attach settlement rule
        rule = settlement_rule or self.default_settlement_rule
        if rule is not None:
            unit.settlement_rule = rule
            logger.info(
                "[5/5] Settlement rule attached  participants=%d",
                len(rule.participants),
            )
        else:
            logger.info("[5/5] No settlement rule provided (skipped)")

        return unit
