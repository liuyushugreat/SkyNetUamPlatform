"""Stateful weak-signal detector for the SkyRescue fault challenge."""

from __future__ import annotations


def detect(records):
    """Yield online anomaly intervals without inspecting labels or fault IDs."""
    active = {}
    for record in records:
        score = 0
        score += record.get("position_residual_m", 0) > 20
        score += record.get("command_latency_ms", 0) > 300
        score += record.get("link_quality", 1.0) < 0.65
        score += record.get("actuator_health") == "degraded"
        score += record.get("reservation_conflict_score", 0) > 0.5
        score += record.get("duplicate_intent_count", 0) > 1
        score += bool(record.get("audit_sequence_gap", False))
        uav, timestamp = record["uav_id"], record["timestamp_s"]
        # A single noisy weak signal is insufficient; structural signals are decisive.
        anomalous = score >= 2 or record.get("duplicate_intent_count", 0) > 1 or record.get("audit_sequence_gap", False)
        if anomalous and uav not in active:
            active[uav] = timestamp
        if not anomalous and uav in active:
            yield {"uav_id": uav, "start_time_s": active.pop(uav), "end_time_s": timestamp}
    for uav, start in active.items():
        yield {"uav_id": uav, "start_time_s": start, "end_time_s": start + 1}
