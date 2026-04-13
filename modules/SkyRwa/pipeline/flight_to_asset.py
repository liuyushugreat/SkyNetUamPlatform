"""End-to-end flight-to-asset pipeline orchestrator.

Connects every layer of the SkyRwa module into a single, sequential pipeline::

    FlightIngestRecord
        → ingest  → FlightAssetUnit (INGESTED)
        → provenance → FlightAssetUnit (EVIDENCE_BUILT)
        → governance → FlightAssetUnit (GOVERNED)
        → valuation → FlightAssetUnit (VALUATED)
        → (optional) settlement rule attachment → SETTLEMENT_READY

Each step is independently replaceable.
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
    """Orchestrates the full ingest → valuation pipeline.

    All components can be injected; defaults are provided for out-of-the-box
    usage.
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
        """Execute the full pipeline and return a fully-populated asset unit."""

        # 1. Ingest
        unit = self.ingestor.ingest(record)
        logger.debug("Ingested → %s  status=%s", unit.asset_unit_id, unit.status)

        # 2. Provenance
        self.evidence_builder.build(unit, record, signer_id=self.signer_id)
        logger.debug("Evidence built → digest=%s", unit.evidence and unit.evidence.digest_hash[:16])

        # 3. Governance
        operator_id = record.operator_id or ""
        self.governance.govern(unit, owner=owner, operator_id=operator_id)
        logger.debug("Governed → class=%s  tradable=%s", unit.asset_class.value, unit.rights_profile and unit.rights_profile.tradable)

        # 4. Valuation
        result = self.valuation.evaluate(unit)
        logger.debug(
            "Valuated → value=%.4f %s  quality=%.4f",
            result.estimated_value,
            result.currency,
            result.quality_score.overall,
        )

        # 5. Attach settlement rule (if any)
        rule = settlement_rule or self.default_settlement_rule
        if rule is not None:
            unit.settlement_rule = rule

        return unit
