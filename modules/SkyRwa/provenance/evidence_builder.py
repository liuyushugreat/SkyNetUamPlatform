"""Evidence builder — constructs a verifiable evidence package from flight data.

The builder collects metadata from a :class:`FlightIngestRecord` and the
partially-populated :class:`FlightAssetUnit`, computes a canonical digest
hash of the package content, and attaches it to the asset unit.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Optional

from ..ingest.flight_ingestor import FlightIngestRecord
from ..models.asset_unit import FlightAssetUnit
from ..models.enums import AssetStatus
from ..models.evidence import (
    EnvironmentContext,
    FlightEvidencePackage,
    MissionResult,
    TelemetrySummary,
)


class EvidenceBuilder:
    """Builds and attaches a :class:`FlightEvidencePackage` to an asset unit."""

    def build(
        self,
        unit: FlightAssetUnit,
        record: FlightIngestRecord,
        *,
        signer_id: Optional[str] = None,
    ) -> FlightAssetUnit:
        """
        Populate ``unit.evidence`` and advance status to ``EVIDENCE_BUILT``.

        Parameters
        ----------
        unit:
            A :class:`FlightAssetUnit` in ``INGESTED`` status.
        record:
            The original ingest record carrying raw flight metadata.
        signer_id:
            Optional identifier of the signing authority.  When provided the
            ``signature`` / ``signed_by`` fields are populated with a
            placeholder — real PKI signing is left to an external adapter.
        """
        duration = 0.0
        if record.start_time and record.end_time:
            duration = (record.end_time - record.start_time).total_seconds()

        telemetry = TelemetrySummary(
            total_points=record.telemetry_points,
            avg_altitude_m=record.avg_altitude_m,
            max_altitude_m=record.max_altitude_m,
            max_speed_mps=record.max_speed_mps,
            avg_speed_mps=record.avg_speed_mps,
            min_battery_pct=record.min_battery_pct,
            avg_battery_pct=record.avg_battery_pct,
            payload_active=record.payload_active,
            trajectory_length_m=record.trajectory_length_m,
        )

        environment = EnvironmentContext(
            weather_condition=record.weather_condition,
            wind_speed_mps=record.wind_speed_mps,
            visibility_km=record.visibility_km,
            temperature_c=record.temperature_c,
            no_fly_zone_incursions=record.no_fly_zone_incursions,
            temporary_restrictions=record.temporary_restrictions,
            risk_events=record.risk_events,
        )

        mission = MissionResult(
            completed=record.mission_completed,
            completion_pct=record.completion_pct,
            deviation_m=record.deviation_m,
            anomalies=record.anomalies,
            alerts=record.alerts,
            violations=record.violations,
        )

        evidence = FlightEvidencePackage(
            flight_id=record.flight_id,
            uav_id=record.uav_id,
            mission_id=record.mission_id,
            operator_id=record.operator_id,
            start_time=record.start_time or datetime.utcnow(),
            end_time=record.end_time or datetime.utcnow(),
            duration_seconds=duration,
            trajectory_hash=record.trajectory_hash,
            raw_data_uri=record.raw_data_uri,
            raw_data_hash=record.raw_data_hash,
            telemetry_summary=telemetry,
            environment=environment,
            mission_result=mission,
        )

        evidence.digest_hash = self._compute_digest(evidence)

        if signer_id:
            evidence.signed_by = signer_id
            evidence.signature = f"placeholder-sig-{evidence.digest_hash[:16]}"
            evidence.signed_at = datetime.utcnow()

        unit.evidence = evidence
        unit.telemetry_hash = evidence.trajectory_hash or evidence.digest_hash
        unit.evidence_uri = evidence.raw_data_uri
        unit.status = AssetStatus.EVIDENCE_BUILT
        unit.updated_at = datetime.utcnow()
        return unit

    # ------------------------------------------------------------------

    @staticmethod
    def _compute_digest(evidence: FlightEvidencePackage) -> str:
        """SHA-256 over the canonical JSON of the evidence payload."""
        payload = evidence.model_dump(
            mode="json",
            exclude={"digest_hash", "signature", "signed_by", "signed_at"},
        )
        canonical = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    @staticmethod
    def verify_digest(evidence: FlightEvidencePackage) -> bool:
        """Re-compute the digest and compare to the stored value."""
        payload = evidence.model_dump(
            mode="json",
            exclude={"digest_hash", "signature", "signed_by", "signed_at"},
        )
        canonical = json.dumps(payload, sort_keys=True, default=str)
        expected = hashlib.sha256(canonical.encode()).hexdigest()
        return expected == evidence.digest_hash
