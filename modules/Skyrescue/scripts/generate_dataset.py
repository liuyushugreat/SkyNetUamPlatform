#!/usr/bin/env python3
"""Generate deterministic synthetic records for SkyRescue-Bench.

The generator uses only the Python standard library so a reviewer can recreate
the dataset from a clean machine. It intentionally produces synthetic records.
"""

import argparse
import hashlib
import json
import math
import random
import shutil
from collections import defaultdict, deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "1.0.0"
GENERATOR_VERSION = "1.0.0"

FACILITIES = [
    {"id": "F001", "name": "North Hospital", "kind": "hospital", "x_km": 0.0, "y_km": 0.0},
    {"id": "F002", "name": "South Hospital", "kind": "hospital", "x_km": 10.0, "y_km": 0.0},
    {"id": "F003", "name": "Emergency Command", "kind": "command", "x_km": 5.0, "y_km": 5.0},
    {"id": "F004", "name": "Logistics Hub", "kind": "logistics", "x_km": 5.0, "y_km": -4.0},
]

TAKEOFF_SITES = [
    ("T001", 1.5, 2.0), ("T002", 3.5, 1.0), ("T003", 7.0, 1.5), ("T004", 8.5, 3.5),
    ("T005", 1.5, -2.5), ("T006", 3.5, -3.5), ("T007", 7.0, -3.0), ("T008", 8.5, -1.5),
]

CORRIDOR_PAIRS = [
    ("F001", "T001"), ("F001", "T002"), ("F002", "T007"), ("F002", "T008"),
    ("F003", "T001"), ("F003", "T003"), ("F003", "T004"), ("F003", "T006"),
    ("F004", "T005"), ("F004", "T006"), ("F004", "T007"), ("T001", "T002"),
    ("T001", "T003"), ("T002", "T003"), ("T002", "T005"), ("T003", "T004"),
    ("T003", "T006"), ("T003", "T007"), ("T004", "T008"), ("T005", "T006"),
    ("T006", "T007"), ("T007", "T008"), ("T005", "F001"), ("T008", "F002"),
]

MISSION_TYPES = {
    "medical_delivery": {"skills": ["medical_payload"], "payload_kg": (0.5, 5.0)},
    "disaster_inspection": {"skills": ["camera", "mapping"], "payload_kg": (0.0, 1.5)},
    "relay": {"skills": ["relay"], "payload_kg": (0.0, 1.0)},
    "casualty_transfer_coordination": {"skills": ["coordination", "medical_payload"], "payload_kg": (1.0, 4.0)},
    "emergency_supply": {"skills": ["cargo"], "payload_kg": (2.0, 12.0)},
}

FAULT_TYPES = {
    "gps_drift": ["position_residual", "route_deviation"],
    "link_delay": ["command_latency_ms", "heartbeat_gap"],
    "mission_replay": ["duplicate_intent", "idempotency_violation"],
    "tool_failure": ["tool_result", "actuator_health"],
    "reservation_conflict": ["reservation_overlap", "safety_gap"],
    "log_tampering": ["hash_mismatch", "sequence_gap"],
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")
    return sha256_bytes(payload.encode("utf-8"))


def write_jsonl(path, records):
    digest = hashlib.sha256()
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            line = canonical(record) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
            count += 1
    return count, digest.hexdigest()


def euclidean(a, b):
    return round(math.hypot(a["x_km"] - b["x_km"], a["y_km"] - b["y_km"]), 3)


def build_scenario():
    nodes = list(FACILITIES)
    nodes.extend({"id": node_id, "name": node_id, "kind": "takeoff_site", "x_km": x, "y_km": y}
                 for node_id, x, y in TAKEOFF_SITES)
    index = {node["id"]: node for node in nodes}
    edges = []
    for number, (source, target) in enumerate(CORRIDOR_PAIRS, start=1):
        edge_id = f"C{number:03d}"
        edges.append({
            "id": edge_id,
            "source": source,
            "target": target,
            "length_km": euclidean(index[source], index[target]),
            "layers_m": [60, 90, 120, 150, 180, 210],
            "max_speed_kmh": 60,
            "min_temporal_gap_s": 20,
        })
    return {
        "nodes": nodes,
        "edges": edges,
        "facilities": FACILITIES,
        "no_fly_zones": [
            {"id": "NFZ001", "kind": "school", "center": [4.0, 2.8], "radius_km": 0.55},
            {"id": "NFZ002", "kind": "industrial_hazard", "center": [6.4, -1.8], "radius_km": 0.75},
        ],
        "operating_constraints": {
            "safety_gap_s": 20,
            "external_action_requires_approval": True,
            "altitude_layers_m": [60, 90, 120, 150, 180, 210],
            "telemetry_hz": 1,
        },
    }


def shortest_path(edges, source, target):
    graph = defaultdict(list)
    for edge in edges:
        graph[edge["source"]].append((edge["target"], edge["id"]))
        graph[edge["target"]].append((edge["source"], edge["id"]))
    queue = deque([(source, [], [])])
    visited = {source}
    while queue:
        current, nodes, corridors = queue.popleft()
        if current == target:
            return [source] + nodes, corridors
        for successor, corridor_id in graph[current]:
            if successor not in visited:
                visited.add(successor)
                queue.append((successor, nodes + [successor], corridors + [corridor_id]))
    raise ValueError(f"No path from {source} to {target}")


def burst_time(rng, duration):
    """Return timestamps with three periods of elevated request intensity."""
    if rng.random() < 0.55:
        windows = [(0.16, 0.24), (0.46, 0.54), (0.74, 0.82)]
        start, end = rng.choice(windows)
        return int(rng.uniform(start * duration, end * duration))
    return int(rng.uniform(0, duration - 1))


def generate_uavs(config, scenario, rng):
    home_sites = [facility["id"] for facility in scenario["facilities"]]
    capabilities = ["medical_payload", "camera", "mapping", "relay", "coordination", "cargo"]
    uavs = []
    for number in range(1, config["uav_count"] + 1):
        selected = rng.sample(capabilities, rng.randint(2, 4))
        uavs.append({
            "uav_id": f"U{number:04d}",
            "home_node": rng.choice(home_sites),
            "max_payload_kg": round(rng.uniform(3.0, 15.0), 1),
            "battery_capacity_wh": rng.choice([900, 1100, 1300, 1500]),
            "cruise_speed_kmh": rng.choice([42, 48, 54, 60]),
            "skills": selected,
            "communication_profile": rng.choice(["5g", "mesh", "hybrid"]),
            "airworthiness": "nominal",
        })
    return uavs


def generate_missions(config, scenario, rng):
    node_ids = [node["id"] for node in scenario["nodes"]]
    missions = []
    for number in range(1, config["task_count"] + 1):
        mission_type = rng.choice(list(MISSION_TYPES))
        origin, destination = rng.sample(node_ids, 2)
        route_nodes, route_corridors = shortest_path(scenario["edges"], origin, destination)
        profile = MISSION_TYPES[mission_type]
        release_time = burst_time(rng, config["duration_seconds"])
        priority = rng.choices([1, 2, 3, 4, 5], weights=[8, 14, 24, 28, 26])[0]
        route_duration = max(180, len(route_corridors) * 150)
        missions.append({
            "mission_id": f"M{number:05d}",
            "request_time_s": release_time,
            "mission_type": mission_type,
            "priority": priority,
            "origin": origin,
            "destination": destination,
            "required_skills": profile["skills"],
            "payload_kg": round(rng.uniform(*profile["payload_kg"]), 2),
            "deadline_s": release_time + route_duration + rng.randint(300, 1200),
            "route_nodes": route_nodes,
            "route_corridors": route_corridors,
            "assigned_layer_m": rng.choice([60, 90, 120, 150, 180, 210]),
            "estimated_duration_s": route_duration,
            "fallback_policy": "reassign_or_safe_hold",
            "evidence_required": ["request", "assignment", "reservation", "execution", "closure"],
        })
    return sorted(missions, key=lambda item: (item["request_time_s"], item["mission_id"]))


def generate_faults(config, missions, uavs, rng):
    fault_count = max(1, round(config["task_count"] * config["fault_rate"]))
    targets = rng.sample(missions, min(fault_count, len(missions)))
    fault_types = list(FAULT_TYPES)
    faults = []
    for number, mission in enumerate(targets, start=1):
        fault_type = rng.choice(fault_types)
        duration = rng.randint(30, 180)
        target_uav = rng.choice(uavs)["uav_id"]
        start_time = min(
            config["duration_seconds"] - duration - 1,
            mission["request_time_s"] + rng.randint(60, max(61, mission["estimated_duration_s"])),
        )
        faults.append({
            "fault_id": f"X{number:04d}",
            "mission_id": mission["mission_id"],
            "target_uav_id": target_uav,
            "fault_type": fault_type,
            "severity": rng.choice(["medium", "high", "critical"]),
            "start_time_s": start_time,
            "end_time_s": start_time + duration,
            "expected_signals": FAULT_TYPES[fault_type],
            "ground_truth_label": "anomalous",
            "recommended_containment": "freeze_external_action_and_revalidate",
        })
    return sorted(faults, key=lambda item: (item["start_time_s"], item["fault_id"]))


def affected_faults(fault_index, uav_id, time_s):
    return [fault for fault in fault_index[uav_id] if fault["start_time_s"] <= time_s < fault["end_time_s"]]


def telemetry_records(config, uavs, faults):
    fault_index = defaultdict(list)
    for fault in faults:
        fault_index[fault["target_uav_id"]].append(fault)
    for time_s in range(config["duration_seconds"]):
        for number, uav in enumerate(uavs, start=1):
            phase = (time_s / 180.0) + number * 0.37
            latitude = 31.1800 + 0.0300 * math.sin(phase)
            longitude = 121.4200 + 0.0350 * math.cos(phase * 0.91)
            battery = max(18.0, 99.0 - (time_s % 2400) * 0.025 - (number % 7) * 0.3)
            active_faults = affected_faults(fault_index, uav["uav_id"], time_s)
            signals = []
            for fault in active_faults:
                signals.extend(fault["expected_signals"])
                if fault["fault_type"] == "gps_drift":
                    latitude += 0.004
                    longitude -= 0.004
                if fault["fault_type"] == "link_delay":
                    command_latency = 1500
                else:
                    command_latency = 85 + (number % 5) * 12
            if not active_faults:
                command_latency = 85 + (number % 5) * 12
            yield {
                "timestamp_s": time_s,
                "uav_id": uav["uav_id"],
                "position": {"latitude": round(latitude, 6), "longitude": round(longitude, 6), "altitude_m": 60 + (number % 6) * 30},
                "velocity_mps": round(10.0 + (number % 8) * 0.55, 2),
                "battery_pct": round(battery, 2),
                "link_quality": 0.25 if active_faults else round(0.92 - (number % 5) * 0.03, 2),
                "command_latency_ms": command_latency,
                "actuator_health": "degraded" if any(f["fault_type"] == "tool_failure" for f in active_faults) else "nominal",
                "anomaly_truth": "anomalous" if active_faults else "nominal",
                "fault_ids": [fault["fault_id"] for fault in active_faults],
            }


def audit_event(sequence, event_type, time_s, subject_id, payload, previous_hash):
    event = {
        "sequence": sequence,
        "event_id": f"E{sequence:07d}",
        "timestamp_s": time_s,
        "event_type": event_type,
        "subject_id": subject_id,
        "payload": payload,
        "previous_hash": previous_hash,
    }
    event["event_hash"] = sha256_bytes(canonical(event).encode("utf-8"))
    return event


def generate_audit_log(missions, faults):
    sequence = 1
    previous_hash = "0" * 64
    events = []
    for mission in missions:
        event = audit_event(sequence, "mission_requested", mission["request_time_s"], mission["mission_id"],
                            {"priority": mission["priority"], "evidence_stage": "request"}, previous_hash)
        events.append(event)
        previous_hash = event["event_hash"]
        sequence += 1
        event = audit_event(sequence, "reservation_committed", mission["request_time_s"] + 5, mission["mission_id"],
                            {"route": mission["route_corridors"], "layer_m": mission["assigned_layer_m"], "evidence_stage": "reservation"}, previous_hash)
        events.append(event)
        previous_hash = event["event_hash"]
        sequence += 1
    for fault in faults:
        event = audit_event(sequence, "fault_injected", fault["start_time_s"], fault["fault_id"],
                            {"mission_id": fault["mission_id"], "fault_type": fault["fault_type"], "evidence_stage": "detection"}, previous_hash)
        events.append(event)
        previous_hash = event["event_hash"]
        sequence += 1
        event = audit_event(sequence, "repair_required", fault["start_time_s"] + 1, fault["mission_id"],
                            {"fault_id": fault["fault_id"], "control_boundary": "approval_required", "evidence_stage": "repair"}, previous_hash)
        events.append(event)
        previous_hash = event["event_hash"]
        sequence += 1
    return sorted(events, key=lambda item: item["sequence"])


def generate(config, output_dir, force):
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"{output_dir} already exists; use --force to replace generated output")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    scenario_dir = output_dir / "scenario"
    scenario_dir.mkdir()
    rng = random.Random(config["random_seed"])
    scenario = build_scenario()
    uavs = generate_uavs(config, scenario, rng)
    missions = generate_missions(config, scenario, rng)
    faults = generate_faults(config, missions, uavs, rng)
    audit_log = generate_audit_log(missions, faults)
    checksums = {}
    checksums["scenario/nodes.json"] = write_json(scenario_dir / "nodes.json", scenario["nodes"])
    checksums["scenario/edges.json"] = write_json(scenario_dir / "edges.json", scenario["edges"])
    checksums["scenario/facilities.json"] = write_json(scenario_dir / "facilities.json", scenario["facilities"])
    checksums["scenario/no_fly_zones.json"] = write_json(scenario_dir / "no_fly_zones.json", scenario["no_fly_zones"])
    checksums["scenario/operating_constraints.json"] = write_json(scenario_dir / "operating_constraints.json", scenario["operating_constraints"])
    counts = {}
    for filename, records in [("missions.jsonl", missions), ("uavs.jsonl", uavs), ("faults.jsonl", faults), ("audit_log.jsonl", audit_log)]:
        count, digest = write_jsonl(output_dir / filename, records)
        counts[filename] = count
        checksums[filename] = digest
    count, digest = write_jsonl(output_dir / "telemetry.jsonl", telemetry_records(config, uavs, faults))
    counts["telemetry.jsonl"] = count
    checksums["telemetry.jsonl"] = digest
    manifest = {
        "dataset_name": "SkyRescue-Bench",
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "synthetic_data": True,
        "generation_timestamp_utc": "2026-08-01T00:00:00Z",
        "configuration": config,
        "record_counts": counts,
        "sha256": checksums,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(f"Generated {config['tier']} dataset at {output_dir}")
    print(json.dumps(counts, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description="Generate a SkyRescue-Bench tier")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tier", choices=["small", "medium", "large", "stress"])
    source.add_argument("--config", type=Path, help="Path to a custom JSON configuration")
    parser.add_argument("--output", type=Path, help="Optional output directory")
    parser.add_argument("--force", action="store_true", help="Replace an existing generated output directory")
    args = parser.parse_args()
    config_path = args.config or ROOT / "configs" / f"{args.tier}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = args.output or ROOT / "data" / args.tier
    generate(config, output_dir, args.force)


if __name__ == "__main__":
    main()
