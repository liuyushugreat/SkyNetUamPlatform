"""Enhanced evaluation with probabilistic mock, single-agent baseline, and audit contrast.

New experiments beyond run_full_eval.py:
  1. Probabilistic mock: LLM verdicts based on ground truth + noise, RAR varies by prompt mode
  2. SingleAgentBaseline: same rules/retrieval/prompt, one agent does everything, no audit
  3. Audit contrast: generic prompt with/without audit+retry under probabilistic mock
  4. Trust protocol detailed case trace
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skygov.agents import ComplianceAgent, RiskAssessmentAgent, ExplanationAgent, AuditAgent
from skygov.agents.base_agent import AgentResult, AgentVerdict, TraceEntry
from skygov.orchestrator import WorkflowEngine, TrustProtocol, TASK_FLIGHT_APPROVAL
from skygov.orchestrator.task_graph import TaskGraph, TaskNode, TaskNodeType
from skygov.utils.metrics import compute_all

SCENARIO_TYPES = {
    "hard_conflict": {
        "wind_range": (6, 9), "battery_range": (5, 14),
        "altitude_range": (250, 500), "max_altitude": 300,
        "visibility_range": (0.5, 3.0), "max_payload_kg": 8.0,
        "speed_range": (15, 30), "max_speed_ms": 20.0,
        "temp_range": (-10, 50), "weight": 0.25,
    },
    "semantic_risk": {
        "wind_range": (4, 6), "battery_range": (15, 40),
        "altitude_range": (200, 320), "max_altitude": 300,
        "visibility_range": (1.0, 5.0), "max_payload_kg": 10.0,
        "speed_range": (10, 22), "max_speed_ms": 20.0,
        "temp_range": (0, 42), "weight": 0.35,
    },
    "safe": {
        "wind_range": (1, 3), "battery_range": (60, 100),
        "altitude_range": (50, 250), "max_altitude": 300,
        "visibility_range": (3.0, 10.0), "max_payload_kg": 10.0,
        "speed_range": (5, 15), "max_speed_ms": 20.0,
        "temp_range": (10, 30), "weight": 0.40,
    },
}

RULE_IDS = [
    "REG-WIND-001", "REG-BAT-001", "REG-ALT-001", "REG-VIS-001",
    "REG-PAYLOAD-001", "REG-SPEED-001", "REG-TEMP-001", "REG-ZONE-001",
]


def compute_ground_truth(telemetry: dict) -> str:
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
    if telemetry.get("current_env_wind", 0) / max(telemetry.get("wind_resistance", 1), 1) > 0.7:
        risk_signals += 1
    if telemetry.get("battery", 100) < 30:
        risk_signals += 1
    if telemetry.get("altitude", 0) / max(telemetry.get("max_altitude", 1), 1) > 0.8:
        risk_signals += 1
    if telemetry.get("visibility_km", 99) < 3.0:
        risk_signals += 1
    if telemetry.get("speed_ms", 0) / max(telemetry.get("max_speed_ms", 1), 1) > 0.8:
        risk_signals += 1
    if risk_signals >= 2:
        return "risk"
    return "safe"


def generate_scenario(idx: int) -> dict:
    r = random.random()
    cumulative = 0.0
    stype, params = "safe", SCENARIO_TYPES["safe"]
    for st, p in SCENARIO_TYPES.items():
        cumulative += p["weight"]
        if r < cumulative:
            stype, params = st, p
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
        "uav_id": f"UAV-E{idx:04d}", "scenario_type": stype, "ground_truth": gt,
        "telemetry": telemetry,
        "mission": {"mission_type": random.choice(["logistics", "surveillance", "emergency"]),
                    "payload_kg": payload_kg,
                    "destination": f"SECTOR-{random.choice('ABCDEF')}{random.randint(1, 20)}"},
        "scenario": {"uav_id": f"UAV-E{idx:04d}", "wind_resistance": wind_res,
                     "current_env_wind": env_wind, "battery": battery, "altitude": altitude,
                     "visibility_km": visibility, "speed_ms": speed, "temperature_c": temperature},
    }


# ---------------------------------------------------------------------------
# Probabilistic mock agents — simulate realistic LLM behaviour
# ---------------------------------------------------------------------------

class ProbabilisticRiskAgent(RiskAssessmentAgent):
    """Risk agent that uses ground truth + noise to simulate real LLM decisions."""

    def __init__(self, risk_correct_rate=0.78, safe_correct_rate=0.85, **kw):
        super().__init__(**kw)
        self.risk_correct_rate = risk_correct_rate
        self.safe_correct_rate = safe_correct_rate

    def execute(self, context):
        gt = context.get("ground_truth", "risk")
        if gt == "risk":
            verdict = AgentVerdict.RISK if random.random() < self.risk_correct_rate else AgentVerdict.SAFE
            conf = round(random.uniform(0.55, 0.85), 2)
        elif gt == "safe":
            verdict = AgentVerdict.SAFE if random.random() < self.safe_correct_rate else AgentVerdict.RISK
            conf = round(random.uniform(0.60, 0.90), 2)
        else:
            verdict = AgentVerdict.RISK
            conf = 0.50
        return AgentResult(
            agent_name=self.name, verdict=verdict, confidence=conf,
            payload={"risk_level": verdict.value, "model": "prob_mock"},
            traces=[TraceEntry(step="prob_risk_eval", source="risk_assessment",
                               detail=f"Probabilistic mock: gt={gt}, verdict={verdict.value}")])


class ProbabilisticExplanationAgent(ExplanationAgent):
    """Explanation agent whose output quality depends on prompt_mode and retry count."""

    def __init__(self, prompt_mode="full", **kw):
        super().__init__(prompt_mode=prompt_mode, **kw)

    def execute(self, context):
        is_retry = context.get("is_retry", False)
        audit_fb = context.get("audit_feedback", {})
        prev_rar = audit_fb.get("rar", 0) if audit_fb else 0

        if self.prompt_mode == "full":
            if random.random() < 0.95:
                n_sentences = random.randint(3, 5)
                cited_rules = random.sample(RULE_IDS, min(n_sentences, len(RULE_IDS)))
                sentences = [f"根据 {r}，该场景存在合规风险。" for r in cited_rules]
                explanation = " ".join(sentences)
                rar_target = 1.0
            else:
                n_sentences = random.randint(3, 5)
                n_cited = n_sentences - 1
                cited_rules = random.sample(RULE_IDS, min(n_cited, len(RULE_IDS)))
                sentences = [f"根据 {r}，该场景存在合规风险。" for r in cited_rules]
                sentences.append("该飞行任务需要进一步审核。")
                random.shuffle(sentences)
                explanation = " ".join(sentences)
                rar_target = n_cited / n_sentences
        elif self.prompt_mode == "generic":
            if not is_retry:
                n_sentences = random.randint(3, 6)
                n_cited = random.randint(0, 1)
                rar_target = n_cited / max(n_sentences, 1)
            elif prev_rar < 0.5:
                n_sentences = random.randint(3, 5)
                n_cited = random.randint(2, 3)
                rar_target = n_cited / max(n_sentences, 1)
            else:
                n_sentences = random.randint(3, 5)
                n_cited = random.randint(3, n_sentences)
                rar_target = n_cited / max(n_sentences, 1)
            cited_rules = random.sample(RULE_IDS, min(n_cited, len(RULE_IDS)))
            sentences = [f"根据 {r}，该场景存在合规风险。" for r in cited_rules]
            for _ in range(n_sentences - n_cited):
                sentences.append("该飞行任务需要进一步审核。")
            random.shuffle(sentences)
            explanation = " ".join(sentences)
        else:
            n_sentences = random.randint(3, 5)
            n_cited = random.randint(1, 3)
            rar_target = n_cited / max(n_sentences, 1)
            cited_rules = random.sample(RULE_IDS, min(n_cited, len(RULE_IDS)))
            sentences = [f"根据 {r}，该场景存在合规风险。" for r in cited_rules]
            for _ in range(n_sentences - n_cited):
                sentences.append("该飞行任务需要进一步审核。")
            random.shuffle(sentences)
            explanation = " ".join(sentences)

        gt = context.get("ground_truth", "risk")
        if gt == "safe" and random.random() < 0.80:
            verdict = AgentVerdict.SAFE
        else:
            verdict = AgentVerdict.RISK
        conf = round(random.uniform(0.55, 0.90), 2)

        return AgentResult(
            agent_name=self.name, verdict=verdict, confidence=conf,
            payload={"explanation": explanation, "cited_rules": cited_rules,
                     "prompt_mode": self.prompt_mode},
            traces=[TraceEntry(step="prob_expl", source="explanation",
                               detail=f"rar_target={rar_target:.2f}, is_retry={is_retry}")])


class SingleAgentGovernor:
    """Single-agent baseline: one agent does compliance + risk + explanation."""

    def __init__(self, prompt_mode="full", risk_cr=0.78, safe_cr=0.85):
        self.comp = ComplianceAgent()
        self.prompt_mode = prompt_mode
        self.risk_cr = risk_cr
        self.safe_cr = safe_cr

    def run(self, scenario):
        comp_result = self.comp.execute(scenario)
        if comp_result.verdict == AgentVerdict.VIOLATION:
            return {"verdict": "violation", "confidence": 1.0,
                    "explanation": "", "cited_rules": [], "retries": 0}

        gt = scenario.get("ground_truth", "risk")
        if gt == "risk":
            verdict = "risk" if random.random() < self.risk_cr else "safe"
        elif gt == "safe":
            verdict = "safe" if random.random() < self.safe_cr else "risk"
        else:
            verdict = "risk"

        if self.prompt_mode == "full":
            rar_target = random.uniform(0.90, 1.0)
        else:
            rar_target = random.uniform(0.05, 0.25)

        n_sent = random.randint(3, 6)
        n_cited = max(0, int(n_sent * rar_target))
        cited = random.sample(RULE_IDS, min(n_cited, len(RULE_IDS)))
        sentences = [f"根据 {r}，该场景存在合规风险。" for r in cited]
        sentences += ["该飞行任务需要进一步审核。"] * (n_sent - n_cited)
        random.shuffle(sentences)

        return {"verdict": verdict, "confidence": round(random.uniform(0.5, 0.9), 2),
                "explanation": " ".join(sentences), "cited_rules": cited, "retries": 0}


def _mean(v): return sum(v) / len(v) if v else 0.0


def normalize_verdict(v):
    if v in ("violation", "veto"): return "violation"
    if v in ("risk", "uncertain"): return "risk"
    return "safe"


def compute_classification_metrics(gt_labels, pred_labels, classes=("violation", "risk", "safe")):
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
        per_class[c] = {"precision": round(prec, 4), "recall": round(rec, 4),
                        "f1": round(f1, 4), "support": sum(cm[c].values())}
    macro_p = _mean([v["precision"] for v in per_class.values()])
    macro_r = _mean([v["recall"] for v in per_class.values()])
    macro_f1 = _mean([v["f1"] for v in per_class.values()])
    return {"accuracy": round(accuracy, 4), "macro_precision": round(macro_p, 4),
            "macro_recall": round(macro_r, 4), "macro_f1": round(macro_f1, 4),
            "per_class": per_class, "confusion_matrix": cm}


def _build_prob_agents(prompt_mode="full", lec_thr=0.2):
    comp = ComplianceAgent()
    risk = ProbabilisticRiskAgent()
    expl = ProbabilisticExplanationAgent(prompt_mode=prompt_mode)
    audit = AuditAgent()
    audit.lec_threshold = lec_thr
    return {"compliance": comp, "risk_assessment": risk, "explanation": expl, "audit": audit}


def experiment_prob_e2e(scenarios, output_dir):
    """Experiment A: End-to-end under probabilistic mock (simulates real API)."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT A: End-to-End (Probabilistic Mock)")
    print("=" * 70)

    agents = _build_prob_agents("full")

    trust = TrustProtocol()
    engine = WorkflowEngine(agents=agents, trust_protocol=trust, max_retries=2, timeout_seconds=180)

    gt_list, pred_list = [], []
    rar_scores = []
    for sc in scenarios:
        res = engine.run(TASK_FLIGHT_APPROVAL, sc)
        pred = normalize_verdict(res["decision"]["final_verdict"])
        gt_list.append(sc["ground_truth"])
        pred_list.append(pred)
        ar = res.get("agent_results", {})
        er = ar.get("explanation", {})
        if er:
            et = er.get("payload", {}).get("explanation", "")
            cr = er.get("payload", {}).get("cited_rules", [])
            m = compute_all(et, cr, set())
            rar_scores.append(m["rar"])

    metrics = compute_classification_metrics(gt_list, pred_list)
    print(f"\n  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Macro-F1:  {metrics['macro_f1']:.4f}")
    print(f"  RAR mean:  {_mean(rar_scores):.4f}")
    for cls, v in metrics["per_class"].items():
        print(f"    {cls:12s}: P={v['precision']:.4f}  R={v['recall']:.4f}  F1={v['f1']:.4f}  n={v['support']}")
    print(f"\n  Confusion Matrix:")
    classes = ("violation", "risk", "safe")
    print(f"    {'':>12s}  {'violation':>10s}  {'risk':>10s}  {'safe':>10s}")
    for g in classes:
        row = "  ".join(f"{metrics['confusion_matrix'][g][p]:>10d}" for p in classes)
        print(f"    {g:>12s}  {row}")

    out = {"metrics": metrics, "rar_mean": round(_mean(rar_scores), 4), "n": len(scenarios)}
    (output_dir / "prob_e2e_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def experiment_single_agent_baseline(scenarios, output_dir):
    """Experiment B: Single-agent baseline comparison."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT B: Single-Agent Baseline vs Multi-Agent SkyGov")
    print("=" * 70)

    all_results = []

    for label, prompt_mode in [("Single-Agent (full)", "full"), ("Single-Agent (generic)", "generic")]:
        baseline = SingleAgentGovernor(prompt_mode=prompt_mode)
        gt_list, pred_list, rars = [], [], []
        for sc in scenarios:
            r = baseline.run(sc)
            pred = normalize_verdict(r["verdict"])
            gt_list.append(sc["ground_truth"])
            pred_list.append(pred)
            m = compute_all(r["explanation"], r["cited_rules"], set())
            rars.append(m["rar"])
        metrics = compute_classification_metrics(gt_list, pred_list)
        print(f"\n  {label}:")
        print(f"    Acc={metrics['accuracy']:.4f}  F1={metrics['macro_f1']:.4f}  RAR={_mean(rars):.4f}")
        all_results.append({"config": label, **metrics, "rar_mean": round(_mean(rars), 4)})

    for label, prompt_mode in [("SkyGov-4Agent (full)", "full"), ("SkyGov-4Agent (generic)", "generic")]:
        agents = _build_prob_agents(prompt_mode)
        trust = TrustProtocol()
        engine = WorkflowEngine(agents=agents, trust_protocol=trust, max_retries=2, timeout_seconds=180)
        gt_list, pred_list, rars, retries_total = [], [], [], 0
        for sc in scenarios:
            res = engine.run(TASK_FLIGHT_APPROVAL, sc)
            pred = normalize_verdict(res["decision"]["final_verdict"])
            gt_list.append(sc["ground_truth"])
            pred_list.append(pred)
            retries_total += res.get("retries", 0)
            ar = res.get("agent_results", {})
            er = ar.get("explanation", {})
            if er:
                et = er.get("payload", {}).get("explanation", "")
                cr = er.get("payload", {}).get("cited_rules", [])
                m = compute_all(et, cr, set())
                rars.append(m["rar"])
        metrics = compute_classification_metrics(gt_list, pred_list)
        print(f"\n  {label}:")
        print(f"    Acc={metrics['accuracy']:.4f}  F1={metrics['macro_f1']:.4f}  RAR={_mean(rars):.4f}  retries={retries_total}")
        all_results.append({"config": label, **metrics, "rar_mean": round(_mean(rars), 4), "retries": retries_total})

    (output_dir / "baseline_comparison.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    return all_results


def experiment_audit_contrast(scenarios, output_dir):
    """Experiment C: Audit before/after contrast under generic prompt."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT C: Audit Contrast (Generic Prompt)")
    print("=" * 70)

    configs = [
        ("No audit, no retry", False, 0),
        ("Audit only (M=0)", True, 0),
        ("Audit + retry (M=1)", True, 1),
        ("Audit + retry (M=2)", True, 2),
    ]

    no_audit_tg = TaskGraph(name="no_audit", description="No audit", entry_node="compliance")
    no_audit_tg.add_node(TaskNode("compliance", TaskNodeType.AGENT, agent_name="compliance", next_nodes=["gate_compliance"]))
    no_audit_tg.add_node(TaskNode("gate_compliance", TaskNodeType.GATE, condition="compliance_verdict", next_nodes=["risk_assessment", "escalate_violation"]))
    no_audit_tg.add_node(TaskNode("risk_assessment", TaskNodeType.AGENT, agent_name="risk_assessment", next_nodes=["explanation"]))
    no_audit_tg.add_node(TaskNode("explanation", TaskNodeType.AGENT, agent_name="explanation", next_nodes=["done"]))
    no_audit_tg.add_node(TaskNode("done", TaskNodeType.MERGE))
    no_audit_tg.add_node(TaskNode("escalate_violation", TaskNodeType.ESCALATE))

    all_results = []
    for label, use_audit, max_retry in configs:
        trust = TrustProtocol()

        if use_audit:
            agents = _build_prob_agents("generic")
            tg = TASK_FLIGHT_APPROVAL
        else:
            comp = ComplianceAgent()
            risk = ProbabilisticRiskAgent()
            expl = ProbabilisticExplanationAgent(prompt_mode="generic")
            agents = {"compliance": comp, "risk_assessment": risk, "explanation": expl}
            tg = no_audit_tg

        engine = WorkflowEngine(agents=agents, trust_protocol=trust,
                                max_retries=max_retry, timeout_seconds=180)

        gt_list, pred_list, rars, ucrs, retries_total = [], [], [], [], 0
        audit_pass_count = 0
        for sc in scenarios:
            res = engine.run(tg, sc)
            pred = normalize_verdict(res["decision"]["final_verdict"])
            gt_list.append(sc["ground_truth"])
            pred_list.append(pred)
            retries_total += res.get("retries", 0)
            ar = res.get("agent_results", {})
            er = ar.get("explanation", {})
            if er:
                et = er.get("payload", {}).get("explanation", "")
                cr = er.get("payload", {}).get("cited_rules", [])
                m = compute_all(et, cr, set())
                rars.append(m["rar"])
                ucrs.append(m["ucr"])
                if m["rar"] >= 0.8 and m["ucr"] <= 0.1:
                    audit_pass_count += 1

        metrics = compute_classification_metrics(gt_list, pred_list)
        n_expl = len(rars) if rars else 1
        pass_rate = round(audit_pass_count / n_expl, 4) if n_expl else 0
        print(f"\n  {label}:")
        print(f"    Acc={metrics['accuracy']:.4f}  F1={metrics['macro_f1']:.4f}")
        print(f"    RAR={_mean(rars):.4f}  UCR={_mean(ucrs):.4f}  PassRate={pass_rate:.4f}  Retries={retries_total}")
        all_results.append({
            "config": label, **metrics,
            "rar_mean": round(_mean(rars), 4), "ucr_mean": round(_mean(ucrs), 4),
            "audit_pass_rate": pass_rate, "retries": retries_total,
        })

    (output_dir / "audit_contrast.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    return all_results


def experiment_trust_protocol_trace(scenarios, output_dir):
    """Experiment D: Detailed trust protocol case traces."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT D: Trust Protocol Case Traces")
    print("=" * 70)

    agents = _build_prob_agents("full")
    trust = TrustProtocol()
    engine = WorkflowEngine(agents=agents, trust_protocol=trust, max_retries=2, timeout_seconds=180)

    cases = []
    type_counts = Counter()
    for sc in scenarios:
        if type_counts[sc["ground_truth"]] >= 2:
            continue
        res = engine.run(TASK_FLIGHT_APPROVAL, sc)
        pred = normalize_verdict(res["decision"]["final_verdict"])
        ar = res.get("agent_results", {})

        case = {
            "uav_id": sc["uav_id"],
            "ground_truth": sc["ground_truth"],
            "predicted": pred,
            "correct": sc["ground_truth"] == pred,
            "telemetry_summary": {
                "wind": f"{sc['telemetry']['current_env_wind']}/{sc['telemetry']['wind_resistance']}",
                "battery": sc["telemetry"]["battery"],
                "altitude": f"{sc['telemetry']['altitude']}/{sc['telemetry']['max_altitude']}",
            },
            "compliance_verdict": ar.get("compliance", {}).get("verdict", "N/A"),
            "risk_verdict": ar.get("risk_assessment", {}).get("verdict", "N/A"),
            "risk_confidence": ar.get("risk_assessment", {}).get("confidence", "N/A"),
            "explanation_verdict": ar.get("explanation", {}).get("verdict", "N/A"),
            "explanation_confidence": ar.get("explanation", {}).get("confidence", "N/A"),
            "audit_verdict": ar.get("audit", {}).get("verdict", "N/A"),
            "retries": res.get("retries", 0),
            "final_verdict": res["decision"]["final_verdict"],
            "trust_detail": res["decision"].get("voting_details", {}),
        }
        cases.append(case)
        type_counts[sc["ground_truth"]] += 1
        print(f"\n  Case {sc['uav_id']} (GT={sc['ground_truth']}):")
        print(f"    Compliance: {case['compliance_verdict']}")
        print(f"    Risk:       {case['risk_verdict']} (conf={case['risk_confidence']})")
        print(f"    Expl:       {case['explanation_verdict']} (conf={case['explanation_confidence']})")
        print(f"    Audit:      {case['audit_verdict']}, retries={case['retries']}")
        print(f"    Final:      {case['final_verdict']} (correct={case['correct']})")

        if all(type_counts[t] >= 2 for t in ("violation", "risk", "safe")):
            break

    (output_dir / "trust_protocol_cases.json").write_text(
        json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
    return cases


def main():
    import logging
    logging.basicConfig(level=logging.ERROR)

    parser = argparse.ArgumentParser(description="SkyGov enhanced evaluation")
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

    experiment_prob_e2e(scenarios, output_dir)
    experiment_single_agent_baseline(scenarios, output_dir)
    experiment_audit_contrast(scenarios, output_dir)
    experiment_trust_protocol_trace(scenarios, output_dir)

    print(f"\n{'=' * 70}")
    print("  ALL ENHANCED EXPERIMENTS COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Outputs in: {output_dir}")


if __name__ == "__main__":
    main()
