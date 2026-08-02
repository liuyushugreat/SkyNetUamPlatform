#!/usr/bin/env python3
"""Generate an independently specified distribution-shift fault challenge.

Unlike FaultChallenge v1.1.0, this generator uses heterogeneous per-UAV
baselines, autoregressive noise, gradual fault envelopes, intermittent
observability, and correlated benign regimes. It deliberately preserves only
the public telemetry schema and fault taxonomy consumed by the frozen
detectors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path


FAULT_TYPES = (
    "gps_drift",
    "link_delay",
    "tool_failure",
    "reservation_conflict",
    "mission_replay",
    "log_tampering",
)


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def smooth_envelope(progress: float) -> float:
    """Raised-cosine envelope with gradual onset and recovery."""
    return 0.5 - 0.5 * math.cos(2.0 * math.pi * clamp(progress, 0.0, 1.0))


def non_overlapping_faults(
    rng: random.Random,
    count: int,
    uavs: int,
    duration: int,
) -> list[dict]:
    by_uav: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    faults: list[dict] = []
    for number in range(count):
        fault_type = FAULT_TYPES[number % len(FAULT_TYPES)]
        for _ in range(2_000):
            uav_id = f"U{rng.randrange(1, uavs + 1):04d}"
            length = rng.randrange(35, 121)
            start = rng.randrange(45, duration - length - 45)
            end = start + length
            if all(end + 20 <= old_start or start >= old_end + 20 for old_start, old_end in by_uav[uav_id]):
                by_uav[uav_id].append((start, end))
                faults.append(
                    {
                        "fault_id": f"XG{number:04d}",
                        "uav_id": uav_id,
                        "fault_type": fault_type,
                        "start_time_s": start,
                        "end_time_s": end,
                        "profile": rng.choice(("gradual", "intermittent", "bursting")),
                    }
                )
                break
        else:
            raise RuntimeError("Unable to place non-overlapping faults")
    return sorted(faults, key=lambda item: (item["start_time_s"], item["uav_id"]))


def is_visible(rng: random.Random, profile: str, progress: float, previous: bool) -> bool:
    envelope = smooth_envelope(progress)
    if profile == "gradual":
        probability = 0.30 + 0.62 * envelope
    elif profile == "bursting":
        probability = 0.82 if previous else 0.48
    else:
        probability = 0.70 if previous else 0.38
    return rng.random() < probability


def add_fault_signal(record: dict, fault: dict, rng: random.Random, visible: bool) -> None:
    if not visible:
        return
    duration = fault["end_time_s"] - fault["start_time_s"]
    progress = (record["timestamp_s"] - fault["start_time_s"]) / max(1, duration - 1)
    envelope = smooth_envelope(progress)
    kind = fault["fault_type"]
    if kind == "gps_drift":
        record["position_residual_m"] += 13.0 + 23.0 * envelope + rng.uniform(-2.0, 3.0)
    elif kind == "link_delay":
        record["command_latency_ms"] += int(125 + 250 * envelope + rng.uniform(-35, 45))
        record["link_quality"] -= 0.08 + 0.25 * envelope + rng.uniform(-0.02, 0.04)
    elif kind == "tool_failure":
        record["actuator_health"] = "degraded"
    elif kind == "reservation_conflict":
        record["reservation_conflict_score"] += 0.28 + 0.52 * envelope
    elif kind == "mission_replay":
        record["duplicate_intent_count"] = 2 + int(envelope > 0.72)
    elif kind == "log_tampering":
        record["audit_sequence_gap"] = True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SkyRescue cross-generator challenge")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261201)
    parser.add_argument("--uavs", type=int, default=20)
    parser.add_argument("--duration", type=int, default=3600)
    parser.add_argument("--faults", type=int, default=120)
    args = parser.parse_args()
    if args.duration < 600:
        parser.error("--duration must be at least 600 seconds")

    rng = random.Random(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    faults = non_overlapping_faults(rng, args.faults, args.uavs, args.duration)
    active_by_uav: defaultdict[str, list[dict]] = defaultdict(list)
    for fault in faults:
        active_by_uav[fault["uav_id"]].append(fault)

    baselines = {
        f"U{number:04d}": {
            "gps": rng.uniform(1.2, 6.0),
            "latency": rng.uniform(70.0, 135.0),
            "link": rng.uniform(0.78, 0.95),
            "gps_state": 0.0,
            "latency_state": 0.0,
            "link_state": 0.0,
        }
        for number in range(1, args.uavs + 1)
    }
    visibility: dict[str, bool] = defaultdict(bool)
    telemetry_path = args.output / "telemetry.jsonl"
    with telemetry_path.open("w", encoding="utf-8") as output:
        for timestamp in range(args.duration):
            weather_regime = math.sin(timestamp / 173.0) + 0.45 * math.sin(timestamp / 41.0)
            correlated_link_event = rng.random() < 0.0025
            correlated_gps_event = rng.random() < 0.0012
            maintenance_event = rng.random() < 0.0008
            for number in range(1, args.uavs + 1):
                uav_id = f"U{number:04d}"
                state = baselines[uav_id]
                state["gps_state"] = 0.88 * state["gps_state"] + rng.gauss(0.0, 0.85)
                state["latency_state"] = 0.82 * state["latency_state"] + rng.gauss(0.0, 10.0)
                state["link_state"] = 0.86 * state["link_state"] + rng.gauss(0.0, 0.018)
                record = {
                    "timestamp_s": timestamp,
                    "uav_id": uav_id,
                    "position_residual_m": state["gps"] + abs(state["gps_state"]) + max(0.0, weather_regime),
                    "command_latency_ms": state["latency"] + state["latency_state"],
                    "link_quality": state["link"] + state["link_state"] - max(0.0, weather_regime) * 0.012,
                    "actuator_health": "nominal",
                    "reservation_conflict_score": clamp(rng.betavariate(1.3, 12.0), 0.0, 0.48),
                    "duplicate_intent_count": 1,
                    "audit_sequence_gap": False,
                }

                active = [
                    fault
                    for fault in active_by_uav[uav_id]
                    if fault["start_time_s"] <= timestamp < fault["end_time_s"]
                ]
                for fault in active:
                    progress = (timestamp - fault["start_time_s"]) / max(
                        1, fault["end_time_s"] - fault["start_time_s"] - 1
                    )
                    visibility[fault["fault_id"]] = is_visible(
                        rng, fault["profile"], progress, visibility[fault["fault_id"]]
                    )
                    add_fault_signal(record, fault, rng, visibility[fault["fault_id"]])

                # Benign correlated regimes differ from the independent point decoys in v1.1.0.
                if not active and correlated_link_event and number % 4 in (0, 1):
                    record["command_latency_ms"] += rng.uniform(190, 350)
                    record["link_quality"] -= rng.uniform(0.18, 0.32)
                if not active and correlated_gps_event and number % 5 == 0:
                    record["position_residual_m"] += rng.uniform(18, 29)
                if not active and maintenance_event and number % 6 == 0:
                    record["actuator_health"] = "degraded"
                if not active and timestamp % 607 in range(4):
                    record["reservation_conflict_score"] = max(
                        record["reservation_conflict_score"], rng.uniform(0.48, 0.68)
                    )
                if not active and timestamp % 911 == number % 11:
                    record["duplicate_intent_count"] = 2
                if not active and timestamp % 997 == number % 13:
                    record["audit_sequence_gap"] = True

                record["position_residual_m"] = round(max(0.0, record["position_residual_m"]), 2)
                record["command_latency_ms"] = int(max(20.0, record["command_latency_ms"]))
                record["link_quality"] = round(clamp(record["link_quality"], 0.20, 0.99), 3)
                record["reservation_conflict_score"] = round(
                    clamp(record["reservation_conflict_score"], 0.0, 0.99), 3
                )
                output.write(json.dumps(record, sort_keys=True) + "\n")

    faults_path = args.output / "faults.jsonl"
    faults_path.write_text(
        "".join(json.dumps(fault, sort_keys=True) + "\n" for fault in faults),
        encoding="utf-8",
    )
    manifest = {
        "name": "SkyRescue-CrossGenerator",
        "version": "1.0.0",
        "synthetic_data": True,
        "seed": args.seed,
        "generator_family": "heterogeneous autoregressive state-space",
        "distribution_shift": [
            "per-UAV heterogeneous baselines",
            "AR(1) temporally correlated noise",
            "raised-cosine gradual fault envelopes",
            "Markov-style intermittent observability",
            "group-correlated benign regimes",
        ],
        "detector_policy": "all detector thresholds are frozen from FaultChallenge v1.1.0",
        "truth_separation": "faults.jsonl is offline scoring only",
        "counts": {"uavs": args.uavs, "duration_s": args.duration, "faults": args.faults},
        "sha256": {"telemetry.jsonl": sha256(telemetry_path), "faults.jsonl": sha256(faults_path)},
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
