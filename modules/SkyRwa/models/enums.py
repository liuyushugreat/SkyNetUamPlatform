"""Enumerations and type constants for the SkyRwa asset pipeline."""

from enum import Enum


# ---------------------------------------------------------------------------
# Asset classification
# ---------------------------------------------------------------------------

class AssetClass(str, Enum):
    """
    Distinguishes the *nature* of a flight-derived data asset.

    - Raw evidence (``FLIGHT_EVIDENCE``) is the unprocessed attestation of a
      single flight.  It is **not** directly tradable — it must go through
      governance before it can become a data product.
    - Governed / derived classes (``RISK_DATASET`` … ``AUDIT_READY_PACKAGE``)
      represent curated data products or service-rights that *may* be priced,
      licensed or settled.
    """
    FLIGHT_EVIDENCE = "flight_evidence"
    RISK_DATASET = "risk_dataset"
    COMPLIANCE_RECORD = "compliance_record"
    MAINTENANCE_SAMPLE = "maintenance_sample"
    ROUTE_OPTIMIZATION_SAMPLE = "route_optimization_sample"
    WEATHER_OPERATION_SAMPLE = "weather_operation_sample"
    AUDIT_READY_PACKAGE = "audit_ready_package"


# ---------------------------------------------------------------------------
# Pipeline status
# ---------------------------------------------------------------------------

class AssetStatus(str, Enum):
    """Lifecycle state of a :class:`FlightAssetUnit` inside the pipeline."""
    INGESTED = "ingested"
    EVIDENCE_BUILT = "evidence_built"
    GOVERNED = "governed"
    VALUATED = "valuated"
    SETTLEMENT_READY = "settlement_ready"
    SETTLED = "settled"
    ARCHIVED = "archived"


# ---------------------------------------------------------------------------
# Data governance
# ---------------------------------------------------------------------------

class DataCategory(str, Enum):
    RAW_TELEMETRY = "raw_telemetry"
    DERIVED_FEATURES = "derived_features"
    AGGREGATED_DATASET = "aggregated_dataset"
    AUDIT_REPORT = "audit_report"
    TRAINING_SAMPLE = "training_sample"


class UsageLevel(str, Enum):
    INTERNAL_ONLY = "internal_only"
    LICENSED_EXTERNAL = "licensed_external"
    TRADABLE_AFTER_DESENSITIZATION = "tradable_after_desensitization"
    NON_TRANSFERABLE = "non_transferable"


# ---------------------------------------------------------------------------
# Revenue & settlement
# ---------------------------------------------------------------------------

class UsageType(str, Enum):
    API_CALL = "api_call"
    SUBSCRIPTION = "subscription"
    TRAINING_USE = "training_use"
    AUDIT_ACCESS = "audit_access"
    BULK_EXPORT = "bulk_export"


class SettlementStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SETTLED = "settled"
    DISPUTED = "disputed"
    CANCELLED = "cancelled"
