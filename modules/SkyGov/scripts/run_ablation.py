"""Ablation & baseline benchmark: runs SkyGov under multiple configurations.

Configurations:
  full_system      — Complete 4-agent pipeline with rule-anchored prompt (8 SPARQL rules)
  generic_prompt   — Full pipeline but ExplanationAgent uses generic prompt (no rules, no format)
  rule_list_only   — Full pipeline, prompt has rule list but no format constraint
  format_only      — Full pipeline, prompt has format constraint but no rule list
  no_audit         — 3-agent pipeline (Compliance+Risk+Explanation), no AuditAgent
  no_compliance    — 3-agent pipeline (Risk+Explanation+Audit), no ComplianceAgent
  rules_3          — Full pipeline but only 3 SPARQL rules (wind, battery, zone)
  rule_engine_llm  — Baseline: ComplianceAgent + single LLM call (no RAG pipeline, no audit)

Scenario parameters reference:
  Wind speed thresholds per CAAC "低空飞行服务保障体系" and ICAO Annex 3.
  Altitude 300m ceiling per PRC "无人驾驶航空器飞行管理暂行条例" Art.25.
  Battery 15% threshold per industry standard (DJI FlySafe; EASA SC-RPAS).
  Visibility 1.5km minimum per CAAC VFR minima (CCAR-91 Section 91.155).
  Temperature -20~45°C per MIL-STD-810 operational envelope.
  Speed 20m/s per typical Class-C UAS operational limits.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skygov.agents import ComplianceAgent, RiskAssessmentAgent, ExplanationAgent, AuditAgent
from skygov.agents.compliance_agent import SPARQL_QUERIES
from skygov.orchestrator import WorkflowEngine, TrustProtocol, TASK_FLIGHT_APPROVAL
from skygov.orchestrator.task_graph import TaskGraph, TaskNode, TaskNodeType
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

SPARQL_3_RULES = {"wind_violation", "battery_critical", "restricted_zone"}


def generate_scenario(idx: int) -> dict:
    r = random.random()
    cumulative = 0.0
    stype = "safe"
    params = SCENARIO_TYPES["safe"]
    for st, p in SCENARIO_TYPES.items():
        cumulative += p["weight"]
        if r < cumulative:
            stype = st
            params = p
            break

    wind_res = random.randint(3, 7)
    env_wind = random.randint(*params["wind_range"])
    battery = random.randint(*params["battery_range"])
    altitude = random.randint(*params["altitude_range"])
    visibility = round(random.uniform(*params["visibility_range"]), 1)
    payload_kg = round(random.uniform(0.5, 10.0), 1)
    speed = round(random.uniform(*params["speed_range"]), 1)
    temperature = round(random.uniform(*params["temp_range"]), 1)

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


def build_no_audit_task_graph() -> TaskGraph:
    """3-agent pipeline without audit: Compliance → Risk → Explanation → done."""
    tg = TaskGraph(name="no_audit", description="No audit pipeline", entry_node="compliance")
    tg.add_node(TaskNode("compliance", TaskNodeType.AGENT, agent_name="compliance", next_nodes=["gate_compliance"]))
    tg.add_node(TaskNode("gate_compliance", TaskNodeType.GATE, condition="compliance_verdict", next_nodes=["risk_assessment", "escalate_violation"]))
    tg.add_node(TaskNode("risk_assessment", TaskNodeType.AGENT, agent_name="risk_assessment", next_nodes=["explanation"]))
    tg.add_node(TaskNode("explanation", TaskNodeType.AGENT, agent_name="explanation", next_nodes=["done"]))
    tg.add_node(TaskNode("done", TaskNodeType.MERGE))
    tg.add_node(TaskNode("escalate_violation", TaskNodeType.ESCALATE))
    return tg


def build_no_compliance_task_graph() -> TaskGraph:
    """3-agent pipeline without compliance: Risk → Explanation → Audit → done."""
    tg = TaskGraph(name="no_compliance", description="No compliance pipeline", entry_node="risk_assessment")
    tg.add_node(TaskNode("risk_assessment", TaskNodeType.AGENT, agent_name="risk_assessment", next_nodes=["explanation"]))
    tg.add_node(TaskNode("explanation", TaskNodeType.AGENT, agent_name="explanation", next_nodes=["audit"]))
    tg.add_node(TaskNode("audit", TaskNodeType.AGENT, agent_name="audit", next_nodes=["done"]))
    tg.add_node(TaskNode("done", TaskNodeType.MERGE))
    return tg


def build_rule_engine_llm_task_graph() -> TaskGraph:
    """Baseline: Compliance → single LLM (risk+explanation in one) → done."""
    tg = TaskGraph(name="rule_engine_llm", description="Baseline: rule engine + LLM", entry_node="compliance")
    tg.add_node(TaskNode("compliance", TaskNodeType.AGENT, agent_name="compliance", next_nodes=["gate_compliance"]))
    tg.add_node(TaskNode("gate_compliance", TaskNodeType.GATE, condition="compliance_verdict", next_nodes=["risk_assessment", "escalate_violation"]))
    tg.add_node(TaskNode("risk_assessment", TaskNodeType.AGENT, agent_name="risk_assessment", next_nodes=["done"]))
    tg.add_node(TaskNode("done", TaskNodeType.MERGE))
    tg.add_node(TaskNode("escalate_violation", TaskNodeType.ESCALATE))
    return tg


class ComplianceAgent3Rules(ComplianceAgent):
    """ComplianceAgent with only 3 SPARQL rules (wind, battery, zone)."""
    def execute(self, context):
        from skygov.agents.base_agent import AgentResult as AR, AgentVerdict as AV, TraceEntry as TE

        self._reset_graph()
        uav_id = context.get("uav_id", "UNKNOWN")
        telemetry = context.get("telemetry", {})
        self.inject_scenario(uav_id, telemetry)

        violations = []
        for rule_key, rule_def in SPARQL_QUERIES.items():
            if rule_key not in SPARQL_3_RULES:
                continue
            query = rule_def["query"].format(ns=self.ns)
            try:
                results = list(self.graph.query(query))
                if results:
                    violations.append(
                        TE(
                            step=f"sparql_{rule_key}",
                            source="compliance_agent",
                            rule_ids=[rule_def["id"]],
                            detail=f"{rule_def['description']} — {len(results)} match(es)",
                        )
                    )
            except Exception as e:
                self.logger.error("SPARQL query %s failed: %s", rule_key, e)

        if violations:
            return AR(agent_name=self.name, verdict=AV.VIOLATION, confidence=1.0,
                      payload={"violation_count": len(violations)}, traces=violations)
        return AR(agent_name=self.name, verdict=AV.SAFE, confidence=1.0,
                  traces=[TE(step="sparql_check_3rules", source="compliance_agent",
                             detail="All 3 rules passed")])


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def _std(vals):
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))


def _ci95(vals):
    """95% confidence interval half-width."""
    n = len(vals)
    if n < 2:
        return 0.0
    return 1.96 * _std(vals) / math.sqrt(n)


def _percentile(vals, p):
    s = sorted(vals)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


CONFIGS = {
    "full_system": {"prompt_mode": "full", "use_audit": True, "use_compliance": True, "sparql_rules": 8, "task_graph": "full"},
    "generic_prompt": {"prompt_mode": "generic", "use_audit": True, "use_compliance": True, "sparql_rules": 8, "task_graph": "full"},
    "rule_list_only": {"prompt_mode": "rule_list_only", "use_audit": True, "use_compliance": True, "sparql_rules": 8, "task_graph": "full"},
    "format_only": {"prompt_mode": "format_only", "use_audit": True, "use_compliance": True, "sparql_rules": 8, "task_graph": "full"},
    "no_audit": {"prompt_mode": "full", "use_audit": False, "use_compliance": True, "sparql_rules": 8, "task_graph": "no_audit"},
    "no_compliance": {"prompt_mode": "full", "use_audit": True, "use_compliance": False, "sparql_rules": 0, "task_graph": "no_compliance"},
    "rules_3": {"prompt_mode": "full", "use_audit": True, "use_compliance": True, "sparql_rules": 3, "task_graph": "full"},
    "rule_engine_llm": {"prompt_mode": "generic", "use_audit": False, "use_compliance": True, "sparql_rules": 8, "task_graph": "rule_engine_llm"},
}


def run_config(config_name: str, scenarios: list, llm_client=None) -> dict:
    cfg = CONFIGS[config_name]
    prompt_mode = cfg["prompt_mode"]

    comp_agent = ComplianceAgent() if cfg["sparql_rules"] == 8 else ComplianceAgent3Rules() if cfg["sparql_rules"] == 3 else None
    risk_agent = RiskAssessmentAgent(llm_client=llm_client)
    expl_agent = ExplanationAgent(llm_client=llm_client, prompt_mode=prompt_mode)
    audit_agent = AuditAgent()

    agents = {}
    if cfg["use_compliance"] and comp_agent:
        agents["compliance"] = comp_agent
    agents["risk_assessment"] = risk_agent
    if cfg["task_graph"] != "rule_engine_llm":
        agents["explanation"] = expl_agent
    if cfg["use_audit"]:
        agents["audit"] = audit_agent

    task_graphs = {
        "full": TASK_FLIGHT_APPROVAL,
        "no_audit": build_no_audit_task_graph(),
        "no_compliance": build_no_compliance_task_graph(),
        "rule_engine_llm": build_rule_engine_llm_task_graph(),
    }
    task_graph = task_graphs[cfg["task_graph"]]

    trust = TrustProtocol()
    max_retries = 2 if cfg["use_audit"] else 0
    engine = WorkflowEngine(agents=agents, trust_protocol=trust, max_retries=max_retries, timeout_seconds=180)

    verdicts = {"safe": 0, "risk": 0, "violation": 0, "uncertain": 0}
    type_verdicts = {st: {"safe": 0, "risk": 0, "violation": 0, "uncertain": 0} for st in SCENARIO_TYPES}
    latencies = []
    rar_scores, lec_scores, ucr_scores = [], [], []
    retry_total = 0
    api_calls = 0
    rows = []

    for scenario in scenarios:
        stype = scenario["scenario_type"]
        t0 = time.perf_counter()
        result = engine.run(task_graph, scenario)
        elapsed = (time.perf_counter() - t0) * 1000

        verdict = result["decision"]["final_verdict"]
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
        type_verdicts[stype][verdict] = type_verdicts[stype].get(verdict, 0) + 1
        latencies.append(elapsed)

        ar = result.get("agent_results", {})
        used_api = llm_client and ("risk_assessment" in ar or "explanation" in ar)
        if used_api:
            api_calls += (1 if "risk_assessment" in ar else 0) + (1 if "explanation" in ar else 0)

        rar_val = lec_val = ucr_val = None
        expl_result = ar.get("explanation", {})
        if expl_result:
            expl_text = expl_result.get("payload", {}).get("explanation", "")
            cited = expl_result.get("payload", {}).get("cited_rules", [])
            metrics = compute_all(expl_text, cited, set())
            rar_val, lec_val, ucr_val = metrics["rar"], metrics["lec"], metrics["ucr"]
            rar_scores.append(rar_val)
            lec_scores.append(lec_val)
            ucr_scores.append(ucr_val)

        retry_total += result.get("retries", 0)

        rows.append({
            "config": config_name,
            "uav_id": scenario["uav_id"],
            "scenario_type": stype,
            "verdict": verdict,
            "latency_ms": round(elapsed, 1),
            "retries": result.get("retries", 0),
            "rar": rar_val, "lec": lec_val, "ucr": ucr_val,
        })

    n = len(scenarios)
    hard_total = sum(type_verdicts["hard_conflict"].values())
    hard_violations = type_verdicts["hard_conflict"].get("violation", 0)
    safe_total = sum(type_verdicts["safe"].values())
    safe_false_positives = type_verdicts["safe"].get("violation", 0)
    sem_total = sum(type_verdicts["semantic_risk"].values())
    sem_violations = type_verdicts["semantic_risk"].get("violation", 0)

    summary = {
        "config": config_name,
        "prompt_mode": cfg["prompt_mode"],
        "sparql_rules": cfg["sparql_rules"],
        "use_audit": cfg["use_audit"],
        "use_compliance": cfg["use_compliance"],
        "n_scenarios": n,
        "verdicts": verdicts,
        "type_verdicts": type_verdicts,
        "hard_rule_interception_rate": round(hard_violations / hard_total, 4) if hard_total else None,
        "safe_false_positive_rate": round(safe_false_positives / safe_total, 4) if safe_total else None,
        "semantic_hard_detection_rate": round(sem_violations / sem_total, 4) if sem_total else None,
        "latency_mean_ms": round(_mean(latencies), 1),
        "latency_std_ms": round(_std(latencies), 1),
        "latency_max_ms": round(max(latencies), 1) if latencies else 0,
        "latency_p95_ms": round(_percentile(latencies, 95), 1) if latencies else 0,
        "rar_mean": round(_mean(rar_scores), 4) if rar_scores else None,
        "rar_std": round(_std(rar_scores), 4) if rar_scores else None,
        "rar_ci95": round(_ci95(rar_scores), 4) if rar_scores else None,
        "lec_mean": round(_mean(lec_scores), 4) if lec_scores else None,
        "ucr_mean": round(_mean(ucr_scores), 4) if ucr_scores else None,
        "ucr_std": round(_std(ucr_scores), 4) if ucr_scores else None,
        "n_evaluated": len(rar_scores),
        "retry_total": retry_total,
        "api_calls": api_calls,
    }
    return {"summary": summary, "rows": rows}


def main():
    import logging
    logging.basicConfig(level=logging.ERROR)

    parser = argparse.ArgumentParser(description="SkyGov ablation & baseline benchmark")
    parser.add_argument("--scenarios", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mock", action="store_true", help="Force mock mode")
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS.keys()),
                        help=f"Configs to run. Available: {list(CONFIGS.keys())}")
    args = parser.parse_args()

    random.seed(args.seed)
    scenarios = [generate_scenario(i) for i in range(args.scenarios)]

    llm_client = None if args.mock else create_llm_client()
    mode_label = "MOCK" if llm_client is None else "REAL API"

    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    all_summaries = []
    all_rows = []

    for cfg_name in args.configs:
        if cfg_name not in CONFIGS:
            print(f"  [SKIP] Unknown config: {cfg_name}")
            continue
        print(f"\n{'=' * 60}")
        print(f"  Running: {cfg_name} ({mode_label}, n={args.scenarios})")
        print(f"{'=' * 60}")

        random.seed(args.seed)
        t0 = time.perf_counter()
        result = run_config(cfg_name, scenarios, llm_client)
        elapsed = time.perf_counter() - t0

        s = result["summary"]
        s["total_time_s"] = round(elapsed, 2)
        all_summaries.append(s)
        all_rows.extend(result["rows"])

        print(f"  Verdicts:  {json.dumps(s['verdicts'])}")
        print(f"  Latency:   mean={s['latency_mean_ms']}ms, p95={s['latency_p95_ms']}ms")
        if s["rar_mean"] is not None:
            print(f"  RAR:       {s['rar_mean']:.4f} ± {s.get('rar_ci95', 0):.4f}")
            print(f"  LEC:       {s['lec_mean']:.4f}")
            print(f"  UCR:       {s['ucr_mean']:.4f}")
        print(f"  Hard-rule interception: {s['hard_rule_interception_rate']}")
        print(f"  Safe FP rate:           {s['safe_false_positive_rate']}")
        print(f"  Time: {elapsed:.1f}s")

    # Save all summaries
    out_summary = output_dir / "ablation_summary.json"
    out_summary.write_text(json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8")

    if all_rows:
        out_csv = output_dir / "ablation_details.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nDetails CSV: {out_csv}")

    # Print comparison table
    print(f"\n{'=' * 100}")
    print("  COMPARISON TABLE")
    print(f"{'=' * 100}")
    header = f"{'Config':<20} {'Rules':>5} {'Audit':>5} {'Prompt':<16} {'Intercept':>9} {'FP':>6} {'RAR':>8} {'LEC':>8} {'UCR':>8} {'Lat(ms)':>10}"
    print(header)
    print("-" * len(header))
    for s in all_summaries:
        print(
            f"{s['config']:<20} "
            f"{s['sparql_rules']:>5} "
            f"{'Y' if s['use_audit'] else 'N':>5} "
            f"{s['prompt_mode']:<16} "
            f"{s['hard_rule_interception_rate'] or '-':>9} "
            f"{s['safe_false_positive_rate'] or '-':>6} "
            f"{s['rar_mean'] if s['rar_mean'] is not None else '-':>8} "
            f"{s['lec_mean'] if s['lec_mean'] is not None else '-':>8} "
            f"{s['ucr_mean'] if s['ucr_mean'] is not None else '-':>8} "
            f"{s['latency_mean_ms']:>10.1f}"
        )

    print(f"\nSummary JSON: {out_summary}")


if __name__ == "__main__":
    main()
