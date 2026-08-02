"""Weak-signal detectors for the SkyRescue fault challenge.

The detectors consume only online telemetry fields.  Fault labels and fault IDs
remain unavailable until the scorer opens ``faults.jsonl`` after inference.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable


DETECTORS = (
    "single_signal",
    "structural_only",
    "persistent_fusion",
    "skyrescue_fusion",
)


FAULT_TYPES = (
    "gps_drift",
    "link_delay",
    "tool_failure",
    "reservation_conflict",
    "mission_replay",
    "log_tampering",
)


def _signals(record: dict) -> dict[str, bool]:
    return {
        "gps_drift": record.get("position_residual_m", 0) > 20,
        "link_delay": record.get("command_latency_ms", 0) > 300
        or record.get("link_quality", 1.0) < 0.65,
        "tool_failure": record.get("actuator_health") == "degraded",
        "reservation_conflict": record.get("reservation_conflict_score", 0) > 0.5,
        "mission_replay": record.get("duplicate_intent_count", 0) > 1,
        "log_tampering": bool(record.get("audit_sequence_gap", False)),
    }


def _strong_signals(record: dict) -> dict[str, bool]:
    return {
        "gps_drift": record.get("position_residual_m", 0) > 26,
        "link_delay": record.get("command_latency_ms", 0) > 380
        and record.get("link_quality", 1.0) < 0.56,
        "tool_failure": record.get("actuator_health") == "degraded",
        "reservation_conflict": record.get("reservation_conflict_score", 0) > 0.55,
        "mission_replay": record.get("duplicate_intent_count", 0) > 1,
        "log_tampering": bool(record.get("audit_sequence_gap", False)),
    }


def _choose_type(signals: dict[str, bool], votes: Counter[str] | None = None) -> str:
    if votes:
        ranked_votes = [item for item in votes.most_common() if item[1] > 0]
        if ranked_votes:
            return ranked_votes[0][0]
    for fault_type in FAULT_TYPES:
        if signals.get(fault_type):
            return fault_type
    return "unknown"


def _decision(record: dict, method: str, streak: int) -> tuple[bool, dict[str, bool], int]:
    signals = _signals(record)
    strong = _strong_signals(record)
    signal_count = sum(signals.values())
    structural = signals["mission_replay"] or signals["log_tampering"]

    if method == "single_signal":
        anomalous = signal_count >= 1
        return anomalous, signals, streak + 1 if anomalous else 0

    if method == "structural_only":
        anomalous = (
            signals["mission_replay"]
            or signals["log_tampering"]
            or signals["tool_failure"]
            or signals["reservation_conflict"]
        )
        return anomalous, signals, streak + 1 if anomalous else 0

    if method == "persistent_fusion":
        weak_anomaly = signal_count >= 1
        next_streak = streak + 1 if weak_anomaly else 0
        anomalous = structural or signal_count >= 2 or next_streak >= 5
        return anomalous, signals, next_streak

    if method == "skyrescue_fusion":
        strong_count = sum(strong.values())
        next_streak = streak + 1 if signal_count >= 1 else 0
        decisive_non_link = any(
            strong[kind]
            for kind in ("gps_drift", "tool_failure", "reservation_conflict", "mission_replay", "log_tampering")
        )
        anomalous = structural or signal_count >= 2 or decisive_non_link or next_streak >= 3
        predicted = strong if strong_count else signals
        return anomalous, predicted, next_streak

    raise ValueError(f"Unknown detector {method!r}; choose from {DETECTORS}")


def detect(records: Iterable[dict], method: str = "skyrescue_fusion"):
    """Yield online anomaly intervals with a predicted fault type.

    Each yielded interval has ``uav_id``, ``start_time_s``, ``end_time_s``,
    ``fault_type``, ``detector``, and ``signals`` fields.
    """

    active: dict[str, dict] = {}
    streaks: defaultdict[str, int] = defaultdict(int)
    for record in records:
        uav_id = record["uav_id"]
        timestamp = int(record["timestamp_s"])
        anomalous, signals, streaks[uav_id] = _decision(record, method, streaks[uav_id])

        if anomalous and uav_id not in active:
            active[uav_id] = {
                "start_time_s": timestamp,
                "votes": Counter(),
                "signals": set(),
            }
        if anomalous:
            active[uav_id]["votes"].update(kind for kind, present in signals.items() if present)
            active[uav_id]["signals"].update(kind for kind, present in signals.items() if present)
        elif uav_id in active:
            state = active.pop(uav_id)
            duration = timestamp - state["start_time_s"]
            if method == "skyrescue_fusion" and duration < 2 and len(state["signals"]) < 2:
                continue
            yield {
                "uav_id": uav_id,
                "start_time_s": state["start_time_s"],
                "end_time_s": timestamp,
                "fault_type": _choose_type(signals, state["votes"]),
                "detector": method,
                "signals": sorted(state["signals"]),
            }

    for uav_id, state in active.items():
        if method == "skyrescue_fusion" and len(state["signals"]) < 2:
            continue
        yield {
            "uav_id": uav_id,
            "start_time_s": state["start_time_s"],
            "end_time_s": state["start_time_s"] + 1,
            "fault_type": _choose_type({}, state["votes"]),
            "detector": method,
            "signals": sorted(state["signals"]),
        }
