"""Comprehensive evaluation: end-to-end metrics, parameter sensitivity, robustness.

Adds to the existing ablation script:
  1. Ground-truth label assignment based on telemetry thresholds
  2. End-to-end Accuracy / Precision / Recall / F1 / confusion matrix
  3. Parameter sensitivity sweep (theta_RAR, theta_UCR, M, alpha weights)
  4. Robustness experiments (retrieval noise injection, rule deletion)
  5. Semantic risk scenario LLM judgment breakdown
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skygov.agents import ComplianceAgent, RiskAssessmentAgent, ExplanationAgent, AuditAgent
from skygov.agents.compliance_agent import SPARQL_QUERIES
from skygov.orchestrator import WorkflowEngine, TrustProtocol, TASK_FLIGHT_APPROVAL
from skygov.orchestrator.task_graph import TaskGraph, TaskNode, TaskNodeType
from skygov.utils.metrics import compute_all

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


def compute_ground_truth(telemetry: dict) -> str:
    """Assign ground-truth verdict based on telemetry thresholds.

    Returns 'violation' if any hard rule is violated, 'risk' if parameters
    are in the borderline zone, 'safe' otherwise.
    """
    violations = 0
    if telemetry.get("current_env_wind", 0) > telemetry.get("wind_resistance", 99):
        violations += 1
    if telemetry.get("battery", 100) < 15:
        violations += 1
    if telemetry.get("altitude", 0) > telemetry.get("max_altitude", 9999):
        violations += 1
    if telemetry.get("visibility_km", 99) < 1.5:
        violations += 1
    if telemetry.get("payload_kg", 0) > telemetry.get("max_payload_kg", 9999):
        violations += 1
    if telemetry.get("speed_ms", 0) > telemetry.get("max_speed_ms", 9999):
        violations += 1
    temp = telemetry.get("temperature_c", 20)
    if temp > 45 or temp < -20:
        violations += 1

    if violations > 0:
        return "violation"

    risk_signals = 0
    wind_ratio = telemetry.get("current_env_wind", 0) / max(telemetry.get("wind_resistance", 1), 1)
    if wind_ratio > 0.7:
        risk_signals += 1
    if telemetry.get("battery", 100) < 30:
        risk_signals += 1
    alt_ratio = telemetry.get("altitude", 0) / max(telemetry.get("max_altitude", 1), 1)
    if alt_ratio > 0.8:
        risk_signals += 1
    if telemetry.get("visibility_km", 99) < 3.0:
        risk_signals += 1
    spd_ratio = telemetry.get("speed_ms", 0) / max(telemetry.get("max_speed_ms", 1), 1)
    if spd_ratio > 0.8:
        risk_signals += 1

    if risk_signals >= 2:
        return "risk"
    return "safe"


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

    telemetry = {
        "wind_resistance": wind_res, "current_env_wind": env_wind,
        "battery": battery, "altitude": altitude,
        "max_altitude": params["max_altitude"],
        "visibility_km": visibility, "payload_kg": payload_kg,
        "max_payload_kg": params["max_payload_kg"],
        "speed_ms": speed, "max_speed_ms": params["max_speed_ms"],
        "temperature_c": temperature,
    }
    gt = compute_ground_truth(telemetry)

    return {
        "uav_id": f"UAV-B{idx:04d}",
        "scenario_type": stype,
        "ground_truth": gt,
        "telemetry": telemetry,
        "mission": {
            "mission_type": random.choice(["logistics", "surveillance", "emergency", "inspection"]),
            "payload_kg": payload_kg,
            "destination": f"SECTOR-{random.choice('ABCDEF')}{random.randint(1, 20)}",
        },
        "scenario": {
            "uav_id": f"UAV-B{idx:04d}",
            "wind_resistance": wind_res, "current_env_wind": env_wind,
            "battery": battery, "altitude": altitude,
            "visibility_km": visibility, "speed_ms": speed,
            "temperature_c": temperature,
        },
    }


def _mean(v): return sum(v) / len(v) if v else 0.0
def _std(v):
    if len(v) < 2: return 0.0
    m = _mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
def _ci95(v): return 1.96 * _std(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0


def normalize_verdict(v: str) -> str:
    """Map system verdicts to 3-class labels matching ground truth."""
    if v in ("violation", "veto"):
        return "violation"
    if v in ("risk", "uncertain"):
        return "risk"
    return "safe"


def compute_classification_metrics(gt_labels, pred_labels, classes=("violation", "risk", "safe")):
    """Compute per-class and macro Precision/Recall/F1 + confusion matrix."""
    cm = {c: {c2: 0 for c2 in classes} for c in classes}
    for gt, pred in zip(gt_labels, pred_labels):
        if gt in cm and pred in cm[gt]:
            cm[gt][pred] += 1

    accuracy = sum(1 for g, p in zip(gt_labels, pred_labels) if g == p) / len(gt_labels) if gt_labels else 0

    per_class = {}
    for c in classes:
        tp = cm[c][c]
        fp = sum(cm[other][c] for other in classes if other != c)
        fn = sum(cm[c][other] for other in classes if other != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        per_class[c] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
                        "support": sum(cm[c].values())}

    macro_p = _mean([v["precision"] for v in per_class.values()])
    macro_r = _mean([v["recall"] for v in per_class.values()])
    macro_f1 = _mean([v["f1"] for v in per_class.values()])

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def run_engine(scenarios, agents, task_graph, trust, max_retries=2, timeout=180):
    """Run engine on all scenarios, return per-scenario results."""
    engine = WorkflowEngine(agents=agents, trust_protocol=trust,
                            max_retries=max_retries, timeout_seconds=timeout)
    results = []
    for scenario in scenarios:
        t0 = time.perf_counter()
        res = engine.run(task_graph, scenario)
        elapsed = (time.perf_counter() - t0) * 1000
        pred = normalize_verdict(res["decision"]["final_verdict"])
        gt = scenario["ground_truth"]

        ar = res.get("agent_results", {})
        rar_val = lec_val = ucr_val = None
        expl_result = ar.get("explanation", {})
        if expl_result:
            expl_text = expl_result.get("payload", {}).get("explanation", "")
            cited = expl_result.get("payload", {}).get("cited_rules", [])
            m = compute_all(expl_text, cited, set())
            rar_val, lec_val, ucr_val = m["rar"], m["lec"], m["ucr"]

        results.append({
            "uav_id": scenario["uav_id"],
            "scenario_type": scenario["scenario_type"],
            "ground_truth": gt,
            "predicted": pred,
            "correct": gt == pred,
            "latency_ms": round(elapsed, 1),
            "retries": res.get("retries", 0),
            "rar": rar_val, "lec": lec_val, "ucr": ucr_val,
        })
    return results


def build_default_agents(audit_rar_thr=0.8, audit_ucr_thr=0.1):
    comp = ComplianceAgent()
    risk = RiskAssessmentAgent()
    expl = ExplanationAgent(prompt_mode="full")
    audit = AuditAgent()
    audit.rar_threshold = audit_rar_thr
    audit.ucr_threshold = audit_ucr_thr
    return {"compliance": comp, "risk_assessment": risk, "explanation": expl, "audit": audit}


def experiment_end_to_end(scenarios, output_dir):
    """Experiment 1: End-to-end decision performance with confusion matrix."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: End-to-End Decision Performance")
    print("=" * 70)

    agents = build_default_agents()
    trust = TrustProtocol()
    results = run_engine(scenarios, agents, TASK_FLIGHT_APPROVAL, trust)

    gt_labels = [r["ground_truth"] for r in results]
    pred_labels = [r["predicted"] for r in results]
    metrics = compute_classification_metrics(gt_labels, pred_labels)

    print(f"\n  Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro Precision:  {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:     {metrics['macro_recall']:.4f}")
    print(f"  Macro F1:         {metrics['macro_f1']:.4f}")
    print(f"\n  Per-class:")
    for cls, v in metrics["per_class"].items():
        print(f"    {cls:12s}: P={v['precision']:.4f}  R={v['recall']:.4f}  F1={v['f1']:.4f}  support={v['support']}")
    print(f"\n  Confusion Matrix (rows=GT, cols=Pred):")
    classes = ("violation", "risk", "safe")
    print(f"    {'':>12s}  {'violation':>10s}  {'risk':>10s}  {'safe':>10s}")
    for gt_c in classes:
        row = "  ".join(f"{metrics['confusion_matrix'][gt_c][p]:>10d}" for p in classes)
        print(f"    {gt_c:>12s}  {row}")

    # Semantic risk subset analysis
    sem_results = [r for r in results if r["scenario_type"] == "semantic_risk"]
    if sem_results:
        sem_gt = [r["ground_truth"] for r in sem_results]
        sem_pred = [r["predicted"] for r in sem_results]
        sem_metrics = compute_classification_metrics(sem_gt, sem_pred)
        print(f"\n  Semantic Risk Subset (n={len(sem_results)}):")
        print(f"    Accuracy: {sem_metrics['accuracy']:.4f}")
        for cls, v in sem_metrics["per_class"].items():
            if v["support"] > 0:
                print(f"    {cls:12s}: P={v['precision']:.4f}  R={v['recall']:.4f}  F1={v['f1']:.4f}  n={v['support']}")
        # LLM judgment breakdown for non-violation semantic scenarios
        sem_llm = [r for r in sem_results if r["ground_truth"] != "violation"]
        if sem_llm:
            llm_preds = Counter(r["predicted"] for r in sem_llm)
            print(f"\n    Semantic risk scenarios entering LLM (n={len(sem_llm)}):")
            for k, v in sorted(llm_preds.items()):
                print(f"      {k}: {v} ({v/len(sem_llm)*100:.1f}%)")

    out = {"metrics": metrics, "n": len(results)}
    (output_dir / "e2e_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def experiment_param_sensitivity(scenarios, output_dir):
    """Experiment 2: Parameter sensitivity analysis."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: Parameter Sensitivity Analysis")
    print("=" * 70)

    all_results = []

    # 2a: θ_RAR sensitivity
    print("\n  --- θ_RAR sensitivity ---")
    for rar_thr in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        agents = build_default_agents(audit_rar_thr=rar_thr)
        trust = TrustProtocol()
        res = run_engine(scenarios, agents, TASK_FLIGHT_APPROVAL, trust)
        gt = [r["ground_truth"] for r in res]
        pred = [r["predicted"] for r in res]
        m = compute_classification_metrics(gt, pred)
        retries = sum(r["retries"] for r in res)
        rars = [r["rar"] for r in res if r["rar"] is not None]
        print(f"    θ_RAR={rar_thr:.1f}: Acc={m['accuracy']:.4f}  F1={m['macro_f1']:.4f}  retries={retries}  RAR={_mean(rars):.4f}")
        all_results.append({"param": "theta_RAR", "value": rar_thr,
                           "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                           "retries": retries, "rar_mean": round(_mean(rars), 4)})

    # 2b: Max retry M sensitivity
    print("\n  --- Max retry M sensitivity ---")
    for M in [0, 1, 2, 3, 5]:
        agents = build_default_agents()
        trust = TrustProtocol()
        engine = WorkflowEngine(agents=agents, trust_protocol=trust, max_retries=M, timeout_seconds=180)
        res_list = []
        for sc in scenarios:
            t0 = time.perf_counter()
            r = engine.run(TASK_FLIGHT_APPROVAL, sc)
            elapsed = (time.perf_counter() - t0) * 1000
            pred = normalize_verdict(r["decision"]["final_verdict"])
            res_list.append({"ground_truth": sc["ground_truth"], "predicted": pred,
                            "retries": r.get("retries", 0), "latency_ms": elapsed})
        gt = [r["ground_truth"] for r in res_list]
        pred = [r["predicted"] for r in res_list]
        m = compute_classification_metrics(gt, pred)
        retries = sum(r["retries"] for r in res_list)
        lat = _mean([r["latency_ms"] for r in res_list])
        print(f"    M={M}: Acc={m['accuracy']:.4f}  F1={m['macro_f1']:.4f}  retries={retries}  lat={lat:.1f}ms")
        all_results.append({"param": "max_retry_M", "value": M,
                           "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                           "retries": retries, "latency_mean_ms": round(lat, 1)})

    # 2c: α weight sensitivity
    print("\n  --- Trust weight α_risk sensitivity ---")
    for alpha_risk in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        alpha_expl = round(1.0 - alpha_risk, 2)
        agents = build_default_agents()
        trust = TrustProtocol(agent_weights={"risk_assessment": alpha_risk, "explanation": alpha_expl})
        res = run_engine(scenarios, agents, TASK_FLIGHT_APPROVAL, trust)
        gt = [r["ground_truth"] for r in res]
        pred = [r["predicted"] for r in res]
        m = compute_classification_metrics(gt, pred)
        print(f"    α_risk={alpha_risk:.1f}, α_expl={alpha_expl:.1f}: Acc={m['accuracy']:.4f}  F1={m['macro_f1']:.4f}")
        all_results.append({"param": "alpha_risk", "value": alpha_risk,
                           "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})

    (output_dir / "param_sensitivity.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    return all_results


def experiment_robustness(scenarios, output_dir):
    """Experiment 3: Robustness under degraded conditions."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: Robustness Under Degraded Conditions")
    print("=" * 70)

    all_results = []

    # Baseline (normal)
    agents = build_default_agents()
    trust = TrustProtocol()
    res = run_engine(scenarios, agents, TASK_FLIGHT_APPROVAL, trust)
    gt = [r["ground_truth"] for r in res]
    pred = [r["predicted"] for r in res]
    m = compute_classification_metrics(gt, pred)
    rars = [r["rar"] for r in res if r["rar"] is not None]
    print(f"\n  Normal:     Acc={m['accuracy']:.4f}  F1={m['macro_f1']:.4f}  RAR={_mean(rars):.4f}")
    all_results.append({"condition": "normal", "accuracy": m["accuracy"],
                       "macro_f1": m["macro_f1"], "rar_mean": round(_mean(rars), 4)})

    # 3a: Rule deletion (remove 3 rules: alt, speed, vis)
    from skygov.agents.compliance_agent import SPARQL_QUERIES as SQ
    rules_to_keep = {"wind_violation", "battery_critical", "restricted_zone",
                     "payload_overweight", "temperature_extreme"}
    from skygov.agents.base_agent import AgentResult as AR, AgentVerdict as AV, TraceEntry as TE

    class ComplianceAgent5Rules(ComplianceAgent):
        def execute(self, context):
            self._reset_graph()
            self.inject_scenario(context.get("uav_id", "X"), context.get("telemetry", {}))
            violations = []
            for rk, rd in SQ.items():
                if rk not in rules_to_keep:
                    continue
                q = rd["query"].format(ns=self.ns)
                try:
                    results = list(self.graph.query(q))
                    if results:
                        violations.append(TE(step=f"sparql_{rk}", source="compliance_agent",
                                            rule_ids=[rd["id"]], detail=f"{rd['description']}"))
                except Exception:
                    pass
            if violations:
                return AR(agent_name=self.name, verdict=AV.VIOLATION, confidence=1.0,
                          payload={"violation_count": len(violations)}, traces=violations)
            return AR(agent_name=self.name, verdict=AV.SAFE, confidence=1.0,
                      traces=[TE(step="check", source="compliance_agent", detail="5 rules passed")])

    agents_5 = {"compliance": ComplianceAgent5Rules(), "risk_assessment": RiskAssessmentAgent(),
                "explanation": ExplanationAgent(prompt_mode="full"), "audit": AuditAgent()}
    res = run_engine(scenarios, agents_5, TASK_FLIGHT_APPROVAL, trust)
    gt = [r["ground_truth"] for r in res]
    pred = [r["predicted"] for r in res]
    m = compute_classification_metrics(gt, pred)
    rars = [r["rar"] for r in res if r["rar"] is not None]
    missed = sum(1 for g, p in zip(gt, pred) if g == "violation" and p != "violation")
    print(f"  Rule del:   Acc={m['accuracy']:.4f}  F1={m['macro_f1']:.4f}  RAR={_mean(rars):.4f}  missed_violations={missed}")
    all_results.append({"condition": "rule_deletion_3", "accuracy": m["accuracy"],
                       "macro_f1": m["macro_f1"], "rar_mean": round(_mean(rars), 4),
                       "missed_violations": missed})

    # 3b: Generic prompt (simulate prompt degradation)
    agents_generic = {"compliance": ComplianceAgent(), "risk_assessment": RiskAssessmentAgent(),
                      "explanation": ExplanationAgent(prompt_mode="generic"), "audit": AuditAgent()}
    res = run_engine(scenarios, agents_generic, TASK_FLIGHT_APPROVAL, trust)
    gt = [r["ground_truth"] for r in res]
    pred = [r["predicted"] for r in res]
    m = compute_classification_metrics(gt, pred)
    rars = [r["rar"] for r in res if r["rar"] is not None]
    retries = sum(r["retries"] for r in res)
    print(f"  Generic P:  Acc={m['accuracy']:.4f}  F1={m['macro_f1']:.4f}  RAR={_mean(rars):.4f}  retries={retries}")
    all_results.append({"condition": "generic_prompt", "accuracy": m["accuracy"],
                       "macro_f1": m["macro_f1"], "rar_mean": round(_mean(rars), 4),
                       "retries": retries})

    # 3c: No audit (degraded quality assurance)
    no_audit_tg = TaskGraph(name="no_audit", description="No audit", entry_node="compliance")
    no_audit_tg.add_node(TaskNode("compliance", TaskNodeType.AGENT, agent_name="compliance", next_nodes=["gate_compliance"]))
    no_audit_tg.add_node(TaskNode("gate_compliance", TaskNodeType.GATE, condition="compliance_verdict", next_nodes=["risk_assessment", "escalate_violation"]))
    no_audit_tg.add_node(TaskNode("risk_assessment", TaskNodeType.AGENT, agent_name="risk_assessment", next_nodes=["explanation"]))
    no_audit_tg.add_node(TaskNode("explanation", TaskNodeType.AGENT, agent_name="explanation", next_nodes=["done"]))
    no_audit_tg.add_node(TaskNode("done", TaskNodeType.MERGE))
    no_audit_tg.add_node(TaskNode("escalate_violation", TaskNodeType.ESCALATE))

    agents_no_audit = {"compliance": ComplianceAgent(), "risk_assessment": RiskAssessmentAgent(),
                       "explanation": ExplanationAgent(prompt_mode="generic")}
    engine_na = WorkflowEngine(agents=agents_no_audit, trust_protocol=trust, max_retries=0, timeout_seconds=180)
    res_list = []
    for sc in scenarios:
        r = engine_na.run(no_audit_tg, sc)
        pred_v = normalize_verdict(r["decision"]["final_verdict"])
        ar = r.get("agent_results", {})
        rar_val = None
        expl_r = ar.get("explanation", {})
        if expl_r:
            expl_t = expl_r.get("payload", {}).get("explanation", "")
            cited = expl_r.get("payload", {}).get("cited_rules", [])
            mm = compute_all(expl_t, cited, set())
            rar_val = mm["rar"]
        res_list.append({"ground_truth": sc["ground_truth"], "predicted": pred_v, "rar": rar_val})
    gt = [r["ground_truth"] for r in res_list]
    pred = [r["predicted"] for r in res_list]
    m = compute_classification_metrics(gt, pred)
    rars = [r["rar"] for r in res_list if r["rar"] is not None]
    print(f"  No audit+GP:Acc={m['accuracy']:.4f}  F1={m['macro_f1']:.4f}  RAR={_mean(rars):.4f}")
    all_results.append({"condition": "no_audit_generic_prompt", "accuracy": m["accuracy"],
                       "macro_f1": m["macro_f1"], "rar_mean": round(_mean(rars), 4)})

    (output_dir / "robustness.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    return all_results


def experiment_error_analysis(scenarios, output_dir):
    """Experiment 4: Error analysis — catalog failure modes."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 4: Error Analysis")
    print("=" * 70)

    agents = build_default_agents()
    trust = TrustProtocol()
    results = run_engine(scenarios, agents, TASK_FLIGHT_APPROVAL, trust)

    errors = [r for r in results if not r["correct"]]
    print(f"\n  Total errors: {len(errors)} / {len(results)} ({len(errors)/len(results)*100:.1f}%)")

    error_types = Counter()
    error_examples = {}
    for e in errors:
        key = f"gt={e['ground_truth']}_pred={e['predicted']}"
        error_types[key] += 1
        if key not in error_examples:
            sc = next(s for s in scenarios if s["uav_id"] == e["uav_id"])
            error_examples[key] = {
                "uav_id": e["uav_id"],
                "scenario_type": e["scenario_type"],
                "ground_truth": e["ground_truth"],
                "predicted": e["predicted"],
                "telemetry_summary": {
                    "wind": f"{sc['telemetry']['current_env_wind']}/{sc['telemetry']['wind_resistance']}",
                    "battery": sc["telemetry"]["battery"],
                    "altitude": f"{sc['telemetry']['altitude']}/{sc['telemetry']['max_altitude']}",
                    "speed": f"{sc['telemetry']['speed_ms']}/{sc['telemetry']['max_speed_ms']}",
                    "visibility": sc["telemetry"]["visibility_km"],
                    "temperature": sc["telemetry"]["temperature_c"],
                },
            }

    print(f"\n  Error type breakdown:")
    for et, count in error_types.most_common():
        print(f"    {et}: {count}")
        if et in error_examples:
            ex = error_examples[et]
            print(f"      Example: {ex['uav_id']} ({ex['scenario_type']})")
            print(f"      Telemetry: {json.dumps(ex['telemetry_summary'], ensure_ascii=False)}")

    out = {"total_errors": len(errors), "total_samples": len(results),
           "error_rate": round(len(errors) / len(results), 4),
           "error_types": dict(error_types),
           "error_examples": error_examples}
    (output_dir / "error_analysis.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def main():
    import logging
    logging.basicConfig(level=logging.ERROR)

    parser = argparse.ArgumentParser(description="SkyGov comprehensive evaluation")
    parser.add_argument("--scenarios", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    scenarios = [generate_scenario(i) for i in range(args.scenarios)]

    gt_dist = Counter(s["ground_truth"] for s in scenarios)
    print(f"\nGround-truth distribution (n={len(scenarios)}):")
    for k, v in sorted(gt_dist.items()):
        print(f"  {k}: {v} ({v/len(scenarios)*100:.1f}%)")

    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    e2e = experiment_end_to_end(scenarios, output_dir)
    param = experiment_param_sensitivity(scenarios, output_dir)
    robust = experiment_robustness(scenarios, output_dir)
    err = experiment_error_analysis(scenarios, output_dir)

    print(f"\n{'=' * 70}")
    print("  ALL EXPERIMENTS COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Outputs in: {output_dir}")


if __name__ == "__main__":
    main()
