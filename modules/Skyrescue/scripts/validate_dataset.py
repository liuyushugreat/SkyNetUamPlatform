#!/usr/bin/env python3
"""Validate record integrity and benchmark invariants for SkyRescue-Bench."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def records(path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_hashes(dataset, manifest):
    for relative_path, expected in manifest["sha256"].items():
        path = dataset / relative_path
        require(path.is_file(), f"Missing file: {relative_path}")
        require(sha256_file(path) == expected, f"Checksum mismatch: {relative_path}")


def check_entities(dataset, config):
    missions = list(records(dataset / "missions.jsonl"))
    uavs = list(records(dataset / "uavs.jsonl"))
    faults = list(records(dataset / "faults.jsonl"))
    require(len(missions) == config["task_count"], "Mission count does not match configuration")
    require(len(uavs) == config["uav_count"], "UAV count does not match configuration")
    mission_ids = {item["mission_id"] for item in missions}
    uav_ids = {item["uav_id"] for item in uavs}
    require(len(mission_ids) == len(missions), "Mission IDs are not unique")
    require(len(uav_ids) == len(uavs), "UAV IDs are not unique")
    for mission in missions:
        require(mission["request_time_s"] < config["duration_seconds"], f"Mission outside duration: {mission['mission_id']}")
        require(mission["route_corridors"], f"Mission has empty route: {mission['mission_id']}")
    for fault in faults:
        require(fault["mission_id"] in mission_ids, f"Unknown mission in fault: {fault['fault_id']}")
        require(fault["target_uav_id"] in uav_ids, f"Unknown UAV in fault: {fault['fault_id']}")
        require(0 <= fault["start_time_s"] < fault["end_time_s"] <= config["duration_seconds"],
                f"Invalid fault duration: {fault['fault_id']}")
    return len(missions), len(uavs), len(faults)


def check_telemetry(dataset, config, uav_count):
    expected_count = config["duration_seconds"] * uav_count
    observed = 0
    for observed, record in enumerate(records(dataset / "telemetry.jsonl"), start=1):
        index = observed - 1
        expected_time = index // uav_count
        expected_uav = f"U{(index % uav_count) + 1:04d}"
        require(record["timestamp_s"] == expected_time, f"Telemetry timestamp out of sequence at row {observed}")
        require(record["uav_id"] == expected_uav, f"Telemetry UAV out of sequence at row {observed}")
        require(record["anomaly_truth"] in {"nominal", "anomalous"}, f"Invalid label at row {observed}")
    require(observed == expected_count, f"Telemetry count is {observed}; expected {expected_count}")


def check_audit(dataset):
    previous_hash = "0" * 64
    count = 0
    for count, event in enumerate(records(dataset / "audit_log.jsonl"), start=1):
        require(event["sequence"] == count, f"Audit sequence gap at event {count}")
        require(event["previous_hash"] == previous_hash, f"Audit chain break at event {count}")
        unsigned = dict(event)
        observed_hash = unsigned.pop("event_hash")
        payload = json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        expected_hash = hashlib.sha256(payload).hexdigest()
        require(observed_hash == expected_hash, f"Audit hash mismatch at event {count}")
        previous_hash = observed_hash
    require(count > 0, "Audit log is empty")
    return count


def main():
    parser = argparse.ArgumentParser(description="Validate SkyRescue-Bench generated data")
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    manifest_path = dataset / "manifest.json"
    require(manifest_path.is_file(), "manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("dataset_name") == "SkyRescue-Bench", "Not a SkyRescue-Bench dataset")
    require(manifest.get("synthetic_data") is True, "Dataset must declare synthetic_data=true")
    config = manifest["configuration"]
    check_hashes(dataset, manifest)
    mission_count, uav_count, fault_count = check_entities(dataset, config)
    check_telemetry(dataset, config, uav_count)
    audit_count = check_audit(dataset)
    print("Validation passed")
    print(json.dumps({"missions": mission_count, "uavs": uav_count, "faults": fault_count, "audit_events": audit_count}, indent=2))


if __name__ == "__main__":
    main()
