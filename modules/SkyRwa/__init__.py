"""
SkyRwa Module
=============

**Flight-to-Asset Pipeline** — transforms every UAM flight into a verifiable,
governable, valuatable data-asset candidate unit.

Architecture layers
-------------------
- ``ingest``       – flight data ingestion
- ``provenance``   – evidence packaging & hashing
- ``rights``       – governance, data-use policy, revenue split
- ``valuation``    – multi-dimensional quality + value scoring
- ``settlement``   – revenue logging, splitting & on-chain adapter
- ``pipeline``     – end-to-end orchestrator
- ``storage``      – lightweight JSON persistence
- ``models``       – pydantic data models shared across layers

Legacy compatibility
--------------------
The original Phase-1 symbols (``DataPacket``, ``ValuationResult``,
``AbstractValuationEngine``, ``PricingEngine``, ``CongestionPricingModel``,
``VoxelParams``, ``PizzaPricingModel``, ``TorusPricingModel``,
``CyclicEmbedding``, ``ArbitrageInjector``, ``calculate_integrity_score``,
``get_betti_numbers``) are still re-exported from this package root for
backward compatibility.
"""

# ── Legacy re-exports (Phase-1 compat) ─────────────────────────────────────
from .valuation_legacy import AbstractValuationEngine, DataPacket, ValuationResult
from .pricing_engine import PricingEngine
from .economics.pricing import CongestionPricingModel, VoxelParams
from .neural_pricing import PizzaPricingModel, TorusPricingModel, CyclicEmbedding
from .adversarial import ArbitrageInjector

try:
    from .topology_metrics import calculate_integrity_score, get_betti_numbers
except ImportError:  # pragma: no cover
    calculate_integrity_score = None  # type: ignore[assignment]
    get_betti_numbers = None  # type: ignore[assignment]

# ── New pipeline exports ────────────────────────────────────────────────────
from .models import (
    AssetClass,
    AssetStatus,
    DataCategory,
    UsageLevel,
    UsageType,
    SettlementStatus,
    FlightEvidencePackage,
    TelemetrySummary,
    EnvironmentContext,
    MissionResult,
    RightsProfile,
    RetentionPolicy,
    RevenueParticipant,
    DataQualityScore,
    AssetValueScore,
    ValuationResultV2,
    SettlementRule,
    RevenueLog,
    SplitEntry,
    FlightAssetUnit,
)
from .ingest import FlightIngestRecord, FlightIngestor
from .provenance import EvidenceBuilder
from .rights import GovernanceEngine
from .valuation import AbstractAssetValuationEngine, RuleBasedValuationEngine
from .settlement import Ledger, RevenueSplitter, OnChainAdapter
from .pipeline import FlightToAssetPipeline
from .storage import JsonStore

__all__ = [
    # legacy
    "AbstractValuationEngine",
    "DataPacket",
    "ValuationResult",
    "PricingEngine",
    "CongestionPricingModel",
    "VoxelParams",
    "PizzaPricingModel",
    "TorusPricingModel",
    "CyclicEmbedding",
    "ArbitrageInjector",
    "calculate_integrity_score",
    "get_betti_numbers",
    # models
    "AssetClass",
    "AssetStatus",
    "DataCategory",
    "UsageLevel",
    "UsageType",
    "SettlementStatus",
    "FlightEvidencePackage",
    "TelemetrySummary",
    "EnvironmentContext",
    "MissionResult",
    "RightsProfile",
    "RetentionPolicy",
    "RevenueParticipant",
    "DataQualityScore",
    "AssetValueScore",
    "ValuationResultV2",
    "SettlementRule",
    "RevenueLog",
    "SplitEntry",
    "FlightAssetUnit",
    # layers
    "FlightIngestRecord",
    "FlightIngestor",
    "EvidenceBuilder",
    "GovernanceEngine",
    "AbstractAssetValuationEngine",
    "RuleBasedValuationEngine",
    "Ledger",
    "RevenueSplitter",
    "OnChainAdapter",
    "FlightToAssetPipeline",
    "JsonStore",
]
