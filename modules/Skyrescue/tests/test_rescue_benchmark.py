"""Tests for the deterministic SkyRescue benchmark runtime."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skyrescue.benchmark import DatasetBundle, detect_observations, evaluate_dataset


def test_observation_detector_does_not_need_labels(tmp_path: Path) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    rows = [
        {"timestamp_s": 0, "uav_id": "U0001", "position": {"latitude": 31.0, "longitude": 121.0}, "link_quality": 0.9, "command_latency_ms": 90, "actuator_health": "nominal", "anomaly_truth": "nominal", "fault_ids": []},
        {"timestamp_s": 1, "uav_id": "U0001", "position": {"latitude": 31.0, "longitude": 121.0}, "link_quality": 0.2, "command_latency_ms": 1600, "actuator_health": "nominal", "anomaly_truth": "anomalous", "fault_ids": ["X0001"]},
        {"timestamp_s": 2, "uav_id": "U0001", "position": {"latitude": 31.0, "longitude": 121.0}, "link_quality": 0.9, "command_latency_ms": 90, "actuator_health": "nominal", "anomaly_truth": "nominal", "fault_ids": []},
    ]
    telemetry.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    intervals, count = detect_observations(telemetry)
    assert count == 3
    assert intervals == {"U0001": [(1, 2)]}


def test_skyrescue_beats_no_audit_on_evidence(tmp_path: Path) -> None:
    (tmp_path / "faults.jsonl").write_text(
        json.dumps({"target_uav_id": "U0001", "start_time_s": 20, "end_time_s": 40}) + "\n",
        encoding="utf-8",
    )
    bundle = DatasetBundle(
        root=tmp_path,
        manifest={"configuration": {"tier": "unit", "duration_seconds": 600}},
        scenario={},
        missions=[{
            "mission_id": "M00001", "priority": 5, "request_time_s": 0,
            "payload_kg": 1.0, "required_skills": ["medical_payload"],
            "route_corridors": ["C001"], "assigned_layer_m": 60,
            "estimated_duration_s": 60, "deadline_s": 300,
        }],
        uavs=[{
            "uav_id": "U0001", "max_payload_kg": 5.0,
            "skills": ["medical_payload"],
        }, {
            "uav_id": "U0002", "max_payload_kg": 5.0,
            "skills": ["medical_payload"],
        }],
        observations={"U0001": [(20, 40)]},
        telemetry_rows=1,
    )
    full = evaluate_dataset(bundle, "skyrescue")
    no_audit = evaluate_dataset(bundle, "no_audit")
    assert full.evidence_completeness > no_audit.evidence_completeness
    assert full.fault_detection_recall is not None
    assert full.invariant_violations == 0
    assert full.duplicate_external_calls == 0
    assert full.residual_reservations == 0
    assert full.replan_p50_ms is not None
    assert full.replan_p99_ms is not None
