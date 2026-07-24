"""Map SkyRwa Pydantic objects to an RDF graph.

Each ``map_*`` method takes a domain object and appends triples to the
supplied :class:`rdflib.Graph`.  The graph can then be serialized as
Turtle, JSON-LD, or any format rdflib supports.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import RDF, XSD

from .namespaces import SKYRWA, SKYRWA_INST, PROV, DCAT, bind_namespaces

from ..models.evidence import FlightEvidencePackage
from ..models.asset_unit import FlightAssetUnit
from ..models.rights import RightsProfile
from ..models.settlement import SettlementRule, RevenueLog, SettlementRecord
from ..models.valuation import ValuationResultV2


def _dt(v: datetime) -> Literal:
    return Literal(v.isoformat(), datatype=XSD.dateTime)


def _iri(prefix: str, local: str) -> URIRef:
    """Build a ``urn:skyrwa:<prefix>:<local>`` IRI."""
    return SKYRWA_INST[f"{prefix}:{local}"]


class SkyRwaMapper:
    """Stateless mapper: domain objects → RDF triples.

    When ``materialize_scoring_context`` is True, each asset candidate
    additionally gets a ``skyrwa:ScoringContext`` node carrying the inputs
    of its compliance/risk scores (mission status, violation/anomaly/NFZ/
    risk-event counts; see ontology/scoring-context-extension.ttl).  This
    lets the threshold and mission-state checks V3/V4/V6 be evaluated by
    the extended SHACL contract (shapes/extended/) instead of only by the
    procedural rule layer, at the cost of extra triples per flight.
    """

    def __init__(self, graph: Optional[Graph] = None, *,
                 materialize_scoring_context: bool = False):
        self.graph = graph if graph is not None else Graph()
        self.materialize_scoring_context = materialize_scoring_context
        bind_namespaces(self.graph)

    # ── FlightEvidencePackage ───────────────────────────────────────────

    def map_evidence(self, ev: FlightEvidencePackage) -> URIRef:
        g = self.graph
        subj = _iri("evidence", ev.evidence_id)
        g.add((subj, RDF.type, SKYRWA.FlightEvidence))
        g.add((subj, SKYRWA.flightId, Literal(ev.flight_id)))
        g.add((subj, SKYRWA.uavId, Literal(ev.uav_id)))
        g.add((subj, SKYRWA.startTime, _dt(ev.start_time)))
        g.add((subj, SKYRWA.endTime, _dt(ev.end_time)))
        g.add((subj, SKYRWA.hasDigest, Literal(ev.digest_hash)))
        if ev.signature:
            g.add((subj, SKYRWA.hasSignature, Literal(ev.signature)))
        if ev.operator_id:
            op = _iri("operator", ev.operator_id)
            g.add((subj, SKYRWA.operatedBy, op))
            g.add((op, RDF.type, SKYRWA.Operator))
        if ev.mission_id:
            mission = _iri("mission", ev.mission_id)
            g.add((subj, SKYRWA.hasMission, mission))
            g.add((mission, RDF.type, SKYRWA.FlightMission))
        uav = _iri("uav", ev.uav_id)
        g.add((subj, SKYRWA.performedByUAV, uav))
        g.add((uav, RDF.type, SKYRWA.UAV))
        g.add((subj, PROV.generatedAtTime, _dt(ev.created_at)))
        return subj

    # ── FlightAssetUnit ─────────────────────────────────────────────────

    def map_asset_unit(self, unit: FlightAssetUnit) -> URIRef:
        g = self.graph
        subj = _iri("asset", unit.asset_unit_id)
        g.add((subj, RDF.type, SKYRWA.AssetCandidate))
        g.add((subj, SKYRWA.flightId, Literal(unit.flight_id)))
        g.add((subj, SKYRWA.uavId, Literal(unit.uav_id)))
        g.add((subj, SKYRWA.hasAssetClass, Literal(unit.asset_class.value)))
        g.add((subj, SKYRWA.hasStatus, Literal(unit.status.value)))
        g.add((subj, SKYRWA.complianceScore, Literal(unit.compliance_score, datatype=XSD.float)))
        g.add((subj, SKYRWA.riskScore, Literal(unit.risk_score, datatype=XSD.float)))
        g.add((subj, SKYRWA.dataQualityScore, Literal(unit.data_quality_score, datatype=XSD.float)))
        if unit.start_time:
            g.add((subj, SKYRWA.startTime, _dt(unit.start_time)))
        if unit.end_time:
            g.add((subj, SKYRWA.endTime, _dt(unit.end_time)))
        if unit.evidence:
            ev_iri = self.map_evidence(unit.evidence)
            g.add((subj, SKYRWA.derivedFromEvidence, ev_iri))
            if self.materialize_scoring_context:
                self._map_scoring_context(subj, unit)
        if unit.rights_profile:
            self._map_rights(subj, unit.rights_profile)
        if unit.valuation_result:
            self._map_valuation(subj, unit.valuation_result)
        if unit.settlement_rule:
            self._map_settlement_rule(subj, unit.settlement_rule)
        for rev in unit.revenue_log:
            self.map_revenue_log(rev)
        return subj

    # ── ScoringContext (optional extension) ─────────────────────────────

    def _map_scoring_context(self, asset_iri: URIRef,
                             unit: FlightAssetUnit) -> URIRef:
        g = self.graph
        mr = unit.evidence.mission_result
        env = unit.evidence.environment
        ctx = _iri("scoring", unit.asset_unit_id)
        g.add((asset_iri, SKYRWA.hasScoringContext, ctx))
        g.add((ctx, RDF.type, SKYRWA.ScoringContext))
        g.add((ctx, SKYRWA.missionCompleted,
               Literal(mr.completed, datatype=XSD.boolean)))
        g.add((ctx, SKYRWA.completionPct,
               Literal(mr.completion_pct, datatype=XSD.float)))
        g.add((ctx, SKYRWA.violationCount,
               Literal(len(mr.violations), datatype=XSD.integer)))
        g.add((ctx, SKYRWA.anomalyCount,
               Literal(len(mr.anomalies), datatype=XSD.integer)))
        g.add((ctx, SKYRWA.nfzIncursionCount,
               Literal(env.no_fly_zone_incursions, datatype=XSD.integer)))
        g.add((ctx, SKYRWA.riskEventCount,
               Literal(len(env.risk_events), datatype=XSD.integer)))
        for v in mr.violations:
            g.add((ctx, SKYRWA.violationLabel, Literal(v)))
        return ctx

    # ── RightsProfile ───────────────────────────────────────────────────

    def _map_rights(self, asset_iri: URIRef, rp: RightsProfile) -> BNode:
        g = self.graph
        node = BNode()
        g.add((asset_iri, SKYRWA.hasRightsProfile, node))
        g.add((node, SKYRWA.isTradable, Literal(rp.tradable, datatype=XSD.boolean)))
        g.add((node, SKYRWA.requiresDesensitization, Literal(rp.desensitization_required, datatype=XSD.boolean)))
        for use in rp.permitted_uses:
            g.add((node, SKYRWA["permittedUse"], Literal(use.value)))
        for use in rp.prohibited_uses:
            g.add((node, SKYRWA["prohibitedUse"], Literal(use)))
        for p in rp.revenue_split:
            share = BNode()
            g.add((node, SKYRWA.hasRevenueShare, share))
            g.add((share, SKYRWA["partyId"], Literal(p.party_id)))
            g.add((share, SKYRWA["role"], Literal(p.role)))
            g.add((share, SKYRWA.sharePct, Literal(p.share_pct, datatype=XSD.float)))
        return node

    def map_rights_profile(self, rp: RightsProfile, asset_unit_id: str = "") -> BNode:
        """Public convenience wrapper."""
        asset_iri = _iri("asset", asset_unit_id) if asset_unit_id else BNode()
        return self._map_rights(asset_iri, rp)

    # ── SettlementRule ──────────────────────────────────────────────────

    def _map_settlement_rule(self, asset_iri: URIRef, rule: SettlementRule) -> BNode:
        g = self.graph
        node = BNode()
        g.add((asset_iri, SKYRWA["hasSettlementRule"], node))
        for p in rule.participants:
            share = BNode()
            g.add((node, SKYRWA.hasRevenueShare, share))
            g.add((share, SKYRWA["partyId"], Literal(p.party_id)))
            g.add((share, SKYRWA["role"], Literal(p.role)))
            g.add((share, SKYRWA.sharePct, Literal(p.share_pct, datatype=XSD.float)))
        return node

    def map_settlement_rule(self, rule: SettlementRule, asset_unit_id: str = "") -> BNode:
        asset_iri = _iri("asset", asset_unit_id) if asset_unit_id else BNode()
        return self._map_settlement_rule(asset_iri, rule)

    # ── RevenueLog ──────────────────────────────────────────────────────

    def map_revenue_log(self, log: RevenueLog) -> URIRef:
        g = self.graph
        subj = _iri("usage", log.usage_id)
        g.add((subj, RDF.type, SKYRWA.UsageEvent))
        g.add((subj, SKYRWA.usageType, Literal(log.usage_type.value)))
        g.add((subj, SKYRWA.grossAmount, Literal(log.gross_amount, datatype=XSD.float)))
        g.add((subj, SKYRWA.consumer, Literal(log.consumer)))
        g.add((subj, PROV.startedAtTime, _dt(log.timestamp)))
        asset_iri = _iri("asset", log.asset_unit_id)
        g.add((asset_iri, SKYRWA.hasUsageEvent, subj))
        if log.consumer:
            consumer_iri = _iri("consumer", log.consumer)
            g.add((subj, SKYRWA.consumedBy, consumer_iri))
            g.add((consumer_iri, RDF.type, SKYRWA.DataConsumer))
        return subj

    # ── SettlementRecord ────────────────────────────────────────────────

    def map_settlement_record(self, rec: SettlementRecord) -> URIRef:
        g = self.graph
        subj = _iri("settlement", rec.settlement_id)
        g.add((subj, RDF.type, SKYRWA.SettlementRecord))
        g.add((subj, SKYRWA.grossAmount, Literal(rec.total_gross, datatype=XSD.float)))
        g.add((subj, PROV.generatedAtTime, _dt(rec.settled_at)))
        asset_iri = _iri("asset", rec.asset_unit_id)
        g.add((asset_iri, SKYRWA.hasSettlementRecord, subj))
        for p in rec.participant_totals:
            share = BNode()
            g.add((subj, SKYRWA.hasRevenueShare, share))
            g.add((share, SKYRWA["partyId"], Literal(p.party_id)))
            g.add((share, SKYRWA["role"], Literal(p.role)))
            g.add((share, SKYRWA.sharePct, Literal(p.share_pct, datatype=XSD.float)))
            g.add((share, SKYRWA["amount"], Literal(p.amount, datatype=XSD.float)))
        return subj

    # ── ValuationResultV2 ───────────────────────────────────────────────

    def _map_valuation(self, asset_iri: URIRef, val: ValuationResultV2) -> BNode:
        g = self.graph
        node = BNode()
        g.add((asset_iri, SKYRWA.hasValuation, node))
        g.add((node, RDF.type, SKYRWA.ValuationExplanation))
        g.add((node, SKYRWA.estimatedValue, Literal(val.estimated_value, datatype=XSD.float)))

        for name, score in val.breakdown.items():
            factor = BNode()
            g.add((node, SKYRWA.hasValuationFactor, factor))
            g.add((factor, RDF.type, SKYRWA.ValuationFactor))
            g.add((factor, SKYRWA["factorName"], Literal(name)))
            g.add((factor, SKYRWA["factorScore"], Literal(score, datatype=XSD.float)))
        return node

    def map_valuation(self, val: ValuationResultV2) -> BNode:
        asset_iri = _iri("asset", val.asset_unit_id)
        return self._map_valuation(asset_iri, val)
