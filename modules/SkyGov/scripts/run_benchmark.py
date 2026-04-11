"""Batch benchmark: evaluates SkyGov on N generated scenarios with real/mock LLM."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skygov.config import SkyGovConfig
from skygov.agents import ComplianceAgent, RiskAssessmentAgent, ExplanationAgent, AuditAgent
from skygov.orchestrator import WorkflowEngine, TrustProtocol, TASK_FLIGHT_APPROVAL
from skygov.utils.metrics import compute_all
from skygov.utils.llm_factory import create_llm_client


SCENARIO_TYPES = {
    "hard_conflict": {
        "wind_range": (6, 9), "battery_range": (5, 14),
        "altitude_range": (250, 500), "max_altitude": 300,
        "visibility_range": (0.5, 3.0),
        "max_payload_kg": 8.0,
        "speed_range": (15, 30), "max_speed_ms": 20.0,
        "temp_range": (-10, 50),
        "weight": 0.25,
    },
    "semantic_risk": {
        "wind_range": (4, 6), "battery_range": (15, 40),
        "altitude_range": (200, 320), "max_altitude": 300,
        "visibility_range": (1.0, 5.0),
        "max_payload_kg": 10.0,
        "speed_range": (10, 22), "max_speed_ms": 20.0,
        "temp_range": (0, 42),
        "weight": 0.35,
    },
    "safe": {
        "wind_range": (1, 3), "battery_range": (60, 100),
        "altitude_range": (50, 250), "max_altitude": 300,
        "visibility_range": (3.0, 10.0),
        "max_payload_kg": 10.0,
        "speed_range": (5, 15), "max_speed_ms": 20.0,
        "temp_range": (10, 30),
        "weight": 0.40,
    },
}


def generate_scenario(idx: int) -> dict:
    """Generate a typed synthetic flight scenario with multi-dimensional telemetry."""
    r = random.random()
    cumulative = 0.0
    for stype, params in SCENARIO_TYPES.items():
        cumulative += params["weight"]
        if r < cumulative:
            break

    wind_lo, wind_hi = params["wind_range"]
    bat_lo, bat_hi = params["battery_range"]
    alt_lo, alt_hi = params["altitude_range"]
    vis_lo, vis_hi = params["visibility_range"]
    spd_lo, spd_hi = params["speed_range"]
    tmp_lo, tmp_hi = params["temp_range"]

    wind_res = random.randint(3, 7)
    env_wind = random.randint(wind_lo, wind_hi)
    battery = random.randint(bat_lo, bat_hi)
    altitude = random.randint(alt_lo, alt_hi)
    visibility = round(random.uniform(vis_lo, vis_hi), 1)
    payload_kg = round(random.uniform(0.5, 10.0), 1)
    speed = round(random.uniform(spd_lo, spd_hi), 1)
    temperature = round(random.uniform(tmp_lo, tmp_hi), 1)

    return {
        "uav_id": f"UAV-B{idx:04d}",
        "scenario_type": stype,
        "telemetry": {
            "wind_resistance": wind_res,
            "current_env_wind": env_wind,
            "battery": battery,
            "altitude": altitude,
            "max_altitude": params["max_altitude"],
            "visibility_km": visibility,
            "payload_kg": payload_kg,
            "max_payload_kg": params["max_payload_kg"],
            "speed_ms": speed,
            "max_speed_ms": params["max_speed_ms"],
            "temperature_c": temperature,
        },
        "mission": {
            "mission_type": random.choice(["logistics", "surveillance", "emergency", "inspection"]),
            "payload_kg": payload_kg,
            "destination": f"SECTOR-{random.choice('ABCDEF')}{random.randint(1, 20)}",
        },
        "scenario": {
            "uav_id": f"UAV-B{idx:04d}",
            "wind_resistance": wind_res,
            "current_env_wind": env_wind,
            "battery": battery,
            "altitude": altitude,
            "visibility_km": visibility,
            "speed_ms": speed,
            "temperature_c": temperature,
        },
    }


def main():
    import logging
    logging.basicConfig(level=logging.ERROR)

    parser = argparse.ArgumentParser(description="SkyGov benchmark evaluation")
    parser.add_argument("--scenarios", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mock", action="store_true", help="Force mock mode (no real API)")
    args = parser.parse_args()

    random.seed(args.seed)

    llm_client = None if args.mock else create_llm_client()
    mode = "MOCK" if llm_client is None else "REAL API"
    print(f"\n[SkyGov Benchmark] Mode: {mode}, Scenarios: {args.scenarios}\n", flush=True)

    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    cfg = SkyGovConfig.from_yaml(config_path) if config_path.exists() else SkyGovConfig()
    agents = {
        "compliance": ComplianceAgent(config=cfg.agents.compliance),
        "risk_assessment": RiskAssessmentAgent(
            config=cfg.agents.risk_assessment, llm_client=llm_client
        ),
        "explanation": ExplanationAgent(
            config=cfg.agents.explanation, llm_client=llm_client
        ),
        "audit": AuditAgent(config=cfg.agents.audit),
    }
    engine = WorkflowEngine(
        agents=agents,
        trust_protocol=TrustProtocol(),
        max_retries=cfg.workflow.max_retries,
        timeout_seconds=cfg.workflow.timeout_seconds,
    )

    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    verdicts = {"safe": 0, "risk": 0, "violation": 0, "uncertain": 0}
    type_verdicts = {st: {"safe": 0, "risk": 0, "violation": 0, "uncertain": 0} for st in SCENARIO_TYPES}
    latencies = []
    rar_scores = []
    lec_scores = []
    ucr_scores = []
    pre_retry_rar_scores = []
    post_retry_rar_scores = []
    retry_count_total = 0
    api_calls = 0
    rows = []

    t_start = time.perf_counter()

    for i in range(args.scenarios):
        scenario = generate_scenario(i)
        stype = scenario["scenario_type"]

        t0 = time.perf_counter()
        result = engine.run(TASK_FLIGHT_APPROVAL, scenario)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        verdict = result["decision"]["final_verdict"]
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
        type_verdicts[stype][verdict] = type_verdicts[stype].get(verdict, 0) + 1
        latencies.append(elapsed_ms)

        rar_val = lec_val = ucr_val = None
        expl_text = ""
        cited = []

        ar = result.get("agent_results", {})
        used_api = "risk_assessment" in ar or "explanation" in ar
        if used_api and llm_client:
            api_calls += (1 if "risk_assessment" in ar else 0) + (1 if "explanation" in ar else 0)

        expl_result = ar.get("explanation", {})
        if expl_result:
            expl_text = expl_result.get("payload", {}).get("explanation", "")
            cited = expl_result.get("payload", {}).get("cited_rules", [])
            metrics = compute_all(expl_text, cited, set())
            rar_val = metrics["rar"]
            lec_val = metrics["lec"]
            ucr_val = metrics["ucr"]
            rar_scores.append(rar_val)
            lec_scores.append(lec_val)
            ucr_scores.append(ucr_val)

        audit_result = ar.get("audit", {})
        audit_passed = audit_result.get("payload", {}).get("passed", None) if audit_result else None

        n_retries = result.get("retries", 0)
        retry_count_total += n_retries
        pre_retry_rar = None
        pre_retry_audits = result.get("pre_retry_audits", [])
        if pre_retry_audits:
            pre_retry_rar = pre_retry_audits[0].get("rar")
            if pre_retry_rar is not None:
                pre_retry_rar_scores.append(pre_retry_rar)
            if rar_val is not None:
                post_retry_rar_scores.append(rar_val)

        rows.append({
            "id": i,
            "uav_id": scenario["uav_id"],
            "scenario_type": stype,
            "wind_res": scenario["telemetry"]["wind_resistance"],
            "env_wind": scenario["telemetry"]["current_env_wind"],
            "battery": scenario["telemetry"]["battery"],
            "altitude": scenario["telemetry"]["altitude"],
            "visibility": scenario["telemetry"]["visibility_km"],
            "speed": scenario["telemetry"]["speed_ms"],
            "temperature": scenario["telemetry"]["temperature_c"],
            "mission_type": scenario["mission"]["mission_type"],
            "verdict": verdict,
            "action": result["decision"]["action"],
            "latency_ms": round(elapsed_ms, 1),
            "retries": n_retries,
            "rar": rar_val,
            "lec": lec_val,
            "ucr": ucr_val,
            "pre_retry_rar": pre_retry_rar,
            "audit_passed": audit_passed,
        })

        if (i + 1) % 20 == 0 or i == args.scenarios - 1:
            elapsed_total = time.perf_counter() - t_start
            print(
                f"  [{i + 1}/{args.scenarios}] "
                f"verdicts={json.dumps(verdicts)} "
                f"avg_lat={sum(latencies)/len(latencies):.0f}ms "
                f"elapsed={elapsed_total:.1f}s",
                flush=True,
            )

    total_time = time.perf_counter() - t_start

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"  SkyGov Benchmark Results: {args.scenarios} scenarios ({mode})")
    print(f"{'=' * 60}")
    print(f"\nOverall verdicts: {json.dumps(verdicts)}")
    print(f"\nBy scenario type:")
    for stype, sv in type_verdicts.items():
        total = sum(sv.values())
        print(f"  {stype:16s}: {json.dumps(sv)} (n={total})")

    print(f"\nLatency:")
    print(f"  Mean: {sum(latencies)/len(latencies):.1f}ms")
    print(f"  Max:  {max(latencies):.1f}ms")
    print(f"  Min:  {min(latencies):.1f}ms")
    print(f"  P95:  {sorted(latencies)[int(len(latencies)*0.95)]:.1f}ms")

    if rar_scores:
        print(f"\nExplanation quality (n={len(rar_scores)} evaluated):")
        print(f"  Mean RAR: {sum(rar_scores)/len(rar_scores):.4f}")
        print(f"  Mean LEC: {sum(lec_scores)/len(lec_scores):.4f}")
        print(f"  Mean UCR: {sum(ucr_scores)/len(ucr_scores):.4f}")

    print(f"\nAudit retries: {retry_count_total} total")
    if pre_retry_rar_scores:
        print(f"  Pre-retry  mean RAR: {sum(pre_retry_rar_scores)/len(pre_retry_rar_scores):.4f} (n={len(pre_retry_rar_scores)})")
    if post_retry_rar_scores:
        print(f"  Post-retry mean RAR: {sum(post_retry_rar_scores)/len(post_retry_rar_scores):.4f} (n={len(post_retry_rar_scores)})")
    if pre_retry_rar_scores and post_retry_rar_scores:
        improvement = sum(post_retry_rar_scores)/len(post_retry_rar_scores) - sum(pre_retry_rar_scores)/len(pre_retry_rar_scores)
        print(f"  RAR improvement:     +{improvement:.4f}")

    if llm_client:
        print(f"\nAPI calls: {api_calls}")
    print(f"Total time: {total_time:.2f}s")

    # ── Save outputs ──
    summary = {
        "mode": mode,
        "n_scenarios": args.scenarios,
        "seed": args.seed,
        "verdicts": verdicts,
        "type_verdicts": type_verdicts,
        "latency_mean_ms": round(sum(latencies) / len(latencies), 1),
        "latency_max_ms": round(max(latencies), 1),
        "latency_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 1),
        "rar_mean": round(sum(rar_scores) / len(rar_scores), 4) if rar_scores else None,
        "lec_mean": round(sum(lec_scores) / len(lec_scores), 4) if lec_scores else None,
        "ucr_mean": round(sum(ucr_scores) / len(ucr_scores), 4) if ucr_scores else None,
        "n_evaluated": len(rar_scores),
        "retry_count_total": retry_count_total,
        "pre_retry_rar_mean": round(sum(pre_retry_rar_scores) / len(pre_retry_rar_scores), 4) if pre_retry_rar_scores else None,
        "post_retry_rar_mean": round(sum(post_retry_rar_scores) / len(post_retry_rar_scores), 4) if post_retry_rar_scores else None,
        "n_retried_scenarios": len(pre_retry_rar_scores),
        "sparql_rule_count": 8,
        "api_calls": api_calls,
        "total_time_s": round(total_time, 2),
    }
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    csv_path = output_dir / "benchmark_details.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nOutputs saved:")
    print(f"  {output_dir / 'benchmark_summary.json'}")
    print(f"  {csv_path}")


if __name__ == "__main__":
    main()
