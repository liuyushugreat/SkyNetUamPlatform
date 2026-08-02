"""Tests for SkyRescue weak-signal fault detectors."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skyrescue.fault_detection import DETECTORS, detect


def test_fault_detectors_emit_typed_intervals() -> None:
    records = [
        {"timestamp_s": 0, "uav_id": "U0001", "position_residual_m": 2, "command_latency_ms": 90, "link_quality": 0.9, "actuator_health": "nominal", "reservation_conflict_score": 0.0, "duplicate_intent_count": 1, "audit_sequence_gap": False},
        {"timestamp_s": 1, "uav_id": "U0001", "position_residual_m": 30, "command_latency_ms": 90, "link_quality": 0.9, "actuator_health": "nominal", "reservation_conflict_score": 0.0, "duplicate_intent_count": 1, "audit_sequence_gap": False},
        {"timestamp_s": 2, "uav_id": "U0001", "position_residual_m": 30, "command_latency_ms": 90, "link_quality": 0.9, "actuator_health": "nominal", "reservation_conflict_score": 0.0, "duplicate_intent_count": 1, "audit_sequence_gap": False},
        {"timestamp_s": 3, "uav_id": "U0001", "position_residual_m": 30, "command_latency_ms": 90, "link_quality": 0.9, "actuator_health": "nominal", "reservation_conflict_score": 0.0, "duplicate_intent_count": 1, "audit_sequence_gap": False},
        {"timestamp_s": 4, "uav_id": "U0001", "position_residual_m": 30, "command_latency_ms": 90, "link_quality": 0.9, "actuator_health": "nominal", "reservation_conflict_score": 0.0, "duplicate_intent_count": 1, "audit_sequence_gap": False},
        {"timestamp_s": 5, "uav_id": "U0001", "position_residual_m": 30, "command_latency_ms": 90, "link_quality": 0.9, "actuator_health": "nominal", "reservation_conflict_score": 0.0, "duplicate_intent_count": 1, "audit_sequence_gap": False},
        {"timestamp_s": 6, "uav_id": "U0001", "position_residual_m": 2, "command_latency_ms": 90, "link_quality": 0.9, "actuator_health": "nominal", "reservation_conflict_score": 0.0, "duplicate_intent_count": 1, "audit_sequence_gap": False},
    ]
    for method in DETECTORS:
        intervals = list(detect(iter(records), method=method))
        if method == "structural_only":
            assert intervals == []
            continue
        assert intervals
        assert intervals[0]["fault_type"] == "gps_drift"
        assert intervals[0]["detector"] == method


def test_causal_detector_rejects_short_reservation_decoy() -> None:
    records = []
    for timestamp in range(12):
        records.append({
            "timestamp_s": timestamp,
            "uav_id": "U0001",
            "position_residual_m": 2,
            "command_latency_ms": 90,
            "link_quality": 0.9,
            "actuator_health": "nominal",
            "reservation_conflict_score": 0.62 if 2 <= timestamp < 6 else 0.1,
            "duplicate_intent_count": 1,
            "audit_sequence_gap": False,
        })
    assert list(detect(iter(records), method="skyrescue_causal")) == []


def test_causal_detector_requires_temporal_reservation_support() -> None:
    records = []
    for timestamp in range(16):
        records.append({
            "timestamp_s": timestamp,
            "uav_id": "U0001",
            "position_residual_m": 2,
            "command_latency_ms": 90,
            "link_quality": 0.9,
            "actuator_health": "nominal",
            "reservation_conflict_score": 0.68 if 2 <= timestamp < 12 else 0.1,
            "duplicate_intent_count": 1,
            "audit_sequence_gap": False,
        })
    intervals = list(detect(iter(records), method="skyrescue_causal"))
    assert len(intervals) == 1
    assert intervals[0]["fault_type"] == "reservation_conflict"
