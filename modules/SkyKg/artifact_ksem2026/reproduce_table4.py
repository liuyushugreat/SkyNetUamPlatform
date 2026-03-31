"""
reproduce_table4.py — Reproduce Table 4: Explanation Quality Evaluation

Paper: SkyKG (KSEM 2026), Section 4.6
Metrics: Rule Alignment Rate (RAR), Label-Explanation Consistency (LEC),
         Unsupported Claim Rate (UCR)

This script samples 100 cases, generates explanations from both
Direct LLM and SkyKG, then automatically scores RAR / LEC / UCR
using a judge-LLM pass. Human annotation (as described in the paper)
would refine these numbers; this script provides the reproducible
automated approximation.

Usage:
    export DEEPSEEK_API_KEY="your_key"
    python reproduce_table4.py
"""

import json
import sys
import os
import random
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

OUTPUT_DIR = current_dir / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(project_root / ".env")
api_key = os.getenv("DEEPSEEK_API_KEY")

SAMPLE_N = 100


def get_llm(temperature=0.1):
    if not api_key:
        return None
    return ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=api_key,
        openai_api_base="https://api.deepseek.com",
        temperature=temperature,
    )


def retrieve_rules(case):
    rules = []
    if "wind_speed" in case.get("environment", {}) and "max_wind_resistance" in case.get("uav", {}):
        rules.append(
            f"RULE #101: Stability Risk exists if environment.wind_speed "
            f"({case['environment']['wind_speed']}) > uav.max_wind_resistance "
            f"({case['uav']['max_wind_resistance']})."
        )
    if case.get("zone", {}).get("is_no_fly"):
        rules.append("RULE #201: Airspace Violation if zone.is_no_fly is True.")
    if not rules:
        rules.append("No specific rule triggered.")
    return rules


def get_explanation_direct_llm(case, llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an aviation safety assistant. Analyze the flight data and "
         "determine the risk level. Output a JSON object with two fields: "
         '"label" (either "High Risk" or "Low Risk") and "explanation" '
         "(a 1-2 sentence justification)."),
        ("user", "Data: {data}"),
    ])
    chain = prompt | llm
    try:
        resp = chain.invoke({"data": json.dumps(case)})
        return resp.content.strip()
    except Exception as e:
        return json.dumps({"label": "Error", "explanation": str(e)})


def get_explanation_skykg(case, llm, rules):
    rules_text = "\n".join(rules)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are SkyKG, a neuro-symbolic reasoner. Use the retrieved "
         "Knowledge Graph rules below to assess risk.\n\n"
         "Retrieved Rules:\n{rules}\n\n"
         "Task: Analyze the flight data. If ANY rule is violated, the label "
         'is "High Risk"; otherwise "Low Risk". Output a JSON object with '
         '"label" and "explanation". In the explanation, explicitly cite '
         "which retrieved rule(s) apply and why."),
        ("user", "Flight Data: {data}"),
    ])
    chain = prompt | llm
    try:
        resp = chain.invoke({"rules": rules_text, "data": json.dumps(case)})
        return resp.content.strip()
    except Exception as e:
        return json.dumps({"label": "Error", "explanation": str(e)})


def judge_explanation(explanation_text, rules, ground_truth, llm_judge):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict explanation-quality judge. You will be given:\n"
         "1. A risk assessment explanation\n"
         "2. The ground-truth rules that apply\n"
         "3. The ground-truth label\n\n"
         "Score the explanation on three binary criteria:\n"
         "- RAR (Rule Alignment): Does the explanation correctly reference "
         "or paraphrase at least one of the ground-truth rules? (1=yes, 0=no)\n"
         "- LEC (Label-Explanation Consistency): Does the reasoning in the "
         "explanation logically support the given label? (1=yes, 0=no)\n"
         "- UCR (Unsupported Claim): Does the explanation contain any factual "
         "assertion NOT grounded in the provided rules or data? (1=yes, 0=no)\n\n"
         'Output ONLY a JSON object: {{"RAR":0or1,"LEC":0or1,"UCR":0or1}}'),
        ("user",
         "Explanation: {explanation}\n"
         "Ground-truth rules: {rules}\n"
         "Ground-truth label: {label}"),
    ])
    chain = prompt | llm_judge
    try:
        resp = chain.invoke({
            "explanation": explanation_text,
            "rules": "\n".join(rules),
            "label": ground_truth,
        })
        content = resp.content.strip()
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(content[start:end])
    except Exception:
        pass
    return {"RAR": 0, "LEC": 0, "UCR": 0}


def main():
    llm = get_llm(temperature=0.1)
    if not llm:
        print("[ERROR] DEEPSEEK_API_KEY not set. Cannot reproduce Table 4.")
        print("  Set it via:  export DEEPSEEK_API_KEY='your_key'")
        sys.exit(1)

    llm_judge = get_llm(temperature=0.0)

    dataset_path = current_dir / "data" / "ksem_large_dataset.json"
    if not dataset_path.exists():
        print("[ERROR] Dataset not found. Run generate_large_scale_dataset.py first.")
        sys.exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        all_cases = json.load(f)

    random.seed(42)
    sample = random.sample(all_cases, min(SAMPLE_N, len(all_cases)))
    print(f"[INFO] Sampled {len(sample)} cases for explanation evaluation.")

    scores = {"Direct LLM": {"RAR": [], "LEC": [], "UCR": []},
              "SkyKG-Ours": {"RAR": [], "LEC": [], "UCR": []}}

    for i, case in enumerate(sample):
        gt = case["ground_truth"]
        rules = retrieve_rules(case)

        expl_llm = get_explanation_direct_llm(case, llm)
        expl_skykg = get_explanation_skykg(case, llm, rules)

        j_llm = judge_explanation(expl_llm, rules, gt, llm_judge)
        j_skykg = judge_explanation(expl_skykg, rules, gt, llm_judge)

        for metric in ("RAR", "LEC", "UCR"):
            scores["Direct LLM"][metric].append(j_llm.get(metric, 0))
            scores["SkyKG-Ours"][metric].append(j_skykg.get(metric, 0))

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(sample)} cases...")

    print("\n" + "=" * 60)
    print("  Table 4: Explanation Quality Evaluation (N={})".format(len(sample)))
    print("=" * 60)
    header = f"{'Method':<15} {'RAR ↑':>8} {'LEC ↑':>8} {'UCR ↓':>8}"
    print(header)
    print("-" * len(header))

    for method in ("Direct LLM", "SkyKG-Ours"):
        rar = sum(scores[method]["RAR"]) / len(scores[method]["RAR"])
        lec = sum(scores[method]["LEC"]) / len(scores[method]["LEC"])
        ucr = sum(scores[method]["UCR"]) / len(scores[method]["UCR"])
        print(f"{method:<15} {rar:>8.2f} {lec:>8.2f} {ucr:>8.2f}")

    results_path = OUTPUT_DIR / "table4_explanation_quality.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    print(f"\nDetailed scores saved to {results_path}")


if __name__ == "__main__":
    main()
