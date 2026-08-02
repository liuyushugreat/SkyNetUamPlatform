#!/usr/bin/env python3
"""Generate a weak-signal, partially observable synthetic fault challenge."""

import argparse
import json
import random
from pathlib import Path

FAULTS = ("gps_drift", "link_delay", "tool_failure", "reservation_conflict", "mission_replay", "log_tampering")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--uavs", type=int, default=20)
    parser.add_argument("--duration", type=int, default=3600)
    parser.add_argument("--faults", type=int, default=120)
    args = parser.parse_args(); rng = random.Random(args.seed); args.output.mkdir(parents=True, exist_ok=True)
    intervals = []
    for number in range(args.faults):
        start = rng.randrange(30, args.duration - 180); duration = rng.randrange(25, 100)
        intervals.append({"fault_id": f"FC{number:04d}", "uav_id": f"U{rng.randrange(1,args.uavs+1):04d}", "fault_type": FAULTS[number % len(FAULTS)], "start_time_s": start, "end_time_s": start + duration, "observable_probability": round(rng.uniform(.55,.82),2)})
    with (args.output / "telemetry.jsonl").open("w", encoding="utf-8") as out:
        for t in range(args.duration):
            for n in range(1, args.uavs + 1):
                uav = f"U{n:04d}"; active = [f for f in intervals if f["uav_id"] == uav and f["start_time_s"] <= t < f["end_time_s"] and rng.random() < f["observable_probability"]]
                record = {"timestamp_s": t, "uav_id": uav, "position_residual_m": round(abs(rng.gauss(3, 2)),1), "command_latency_ms": int(max(20, rng.gauss(95, 25))), "link_quality": round(min(.99,max(.45,rng.gauss(.88,.06))),2), "actuator_health": "nominal", "reservation_conflict_score": round(max(0,rng.gauss(.08,.08)),2), "duplicate_intent_count": 1, "audit_sequence_gap": False}
                for fault in active:
                    kind = fault["fault_type"]
                    if kind == "gps_drift": record["position_residual_m"] += rng.uniform(16, 32)
                    elif kind == "link_delay": record["command_latency_ms"] += rng.randrange(180, 360); record["link_quality"] = round(max(.45, record["link_quality"] - rng.uniform(.12,.28)),2)
                    elif kind == "tool_failure": record["actuator_health"] = "degraded" if rng.random() < .65 else "nominal"
                    elif kind == "reservation_conflict": record["reservation_conflict_score"] = round(rng.uniform(.42,.82),2)
                    elif kind == "mission_replay": record["duplicate_intent_count"] = 2
                    elif kind == "log_tampering": record["audit_sequence_gap"] = True
                # Correlated but benign telemetry bursts create realistic false alarms.
                if not active and rng.random() < .007:
                    record["command_latency_ms"] = rng.randrange(310, 430)
                    record["link_quality"] = round(rng.uniform(.52, .64), 2)
                if not active and rng.random() < .002:
                    record["position_residual_m"] = round(rng.uniform(21, 31), 1)
                if not active and rng.random() < .0015:
                    record["actuator_health"] = "degraded"
                if not active and rng.random() < .0015:
                    record["reservation_conflict_score"] = round(rng.uniform(.52, .76), 2)
                if not active and rng.random() < .0006:
                    record["duplicate_intent_count"] = 2
                if not active and rng.random() < .0006:
                    record["audit_sequence_gap"] = True
                out.write(json.dumps(record, sort_keys=True) + "\n")
    (args.output / "faults.jsonl").write_text("".join(json.dumps(f, sort_keys=True) + "\n" for f in intervals), encoding="utf-8")
    (args.output / "manifest.json").write_text(json.dumps({"name":"SkyRescue-FaultChallenge","version":"1.1.0","synthetic_data":True,"seed":args.seed,"background_decoys":["gps_residual","link_burst","actuator_degraded","reservation_score","duplicate_intent","audit_gap"],"truth_separation":"faults.jsonl is offline scoring only"}, indent=2) + "\n", encoding="utf-8")
    print(args.output)
if __name__ == "__main__": main()
