"""
reproduce_table3.py — Reproduce Table 3: Robustness to Sensor Noise

Paper: SkyKG (KSEM 2026), Section 4.3
Ablation: inject additional Gaussian noise at three severity levels
(sigma = 0.05, 0.15, 0.30 relative to telemetry magnitude) and
measure accuracy degradation for all three methods.

Usage:
    export DEEPSEEK_API_KEY="your_key"
    python reproduce_table3.py
"""

import json
import copy
import sys
import os
import time
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from sklearn.metrics import accuracy_score

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

OUTPUT_DIR = current_dir / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(project_root / ".env")
api_key = os.getenv("DEEPSEEK_API_KEY")

NOISE_LEVELS = {"Mild": 0.05, "Moderate": 0.15, "Severe": 0.30}


def get_llm():
    if not api_key:
        return None
    return ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=api_key,
        openai_api_base="https://api.deepseek.com",
        temperature=0.1,
    )


def inject_noise(case, sigma_rel):
    """Add relative Gaussian noise to numerical telemetry fields."""
    noisy = copy.deepcopy(case)
    ws = noisy["environment"].get("wind_speed", 0)
    if ws > 0:
        noisy["environment"]["wind_speed"] = round(
            max(0.0, ws + np.random.normal(0, sigma_rel * ws)), 2
        )
    res = noisy["uav"].get("max_wind_resistance", 0)
    if res > 0:
        noisy["uav"]["max_wind_resistance"] = round(
            max(1.0, res + np.random.normal(0, sigma_rel * res)), 2
        )
    bat = noisy["uav"].get("battery", 50)
    noisy["uav"]["battery"] = round(
        max(0.0, min(100.0, bat + np.random.normal(0, sigma_rel * bat))), 1
    )
    return noisy


def run_baseline_rule(case):
    zone = case.get("zone", {})
    if zone.get("is_no_fly"):
        return "High Risk"
    return "Low Risk"


def run_baseline_llm(case, llm):
    if not llm:
        return "Error"
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an aviation safety assistant. Analyze the input JSON "
         "and determine if the risk is 'High Risk' or 'Low Risk'. "
         "Output ONLY the label."),
        ("user", "Data: {data}"),
    ])
    chain = prompt | llm
    try:
        resp = chain.invoke({"data": json.dumps(case)})
        content = resp.content.strip()
        if "High Risk" in content:
            return "High Risk"
        return "Low Risk"
    except Exception:
        return "Error"


def run_skykg(case, llm):
    if not llm:
        return "Error"
    rules = []
    env = case.get("environment", {})
    uav = case.get("uav", {})
    if "wind_speed" in env and "max_wind_resistance" in uav:
        rules.append(
            "RULE: Stability Risk exists if environment.wind_speed "
            "> uav.max_wind_resistance."
        )
    if case.get("zone", {}).get("is_no_fly"):
        rules.append("RULE: Airspace Violation if zone.is_no_fly is True.")
    rules_text = "\n".join(rules) if rules else "No specific rule triggered."

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are SkyKG, a neuro-symbolic reasoner. Use the retrieved "
         "Knowledge Graph rules to assess risk.\n"
         "Knowledge:\n{rules}\n\n"
         "Task: Analyze the flight data. If ANY rule is violated, "
         "return 'High Risk'. Otherwise 'Low Risk'. Output ONLY the label."),
        ("user", "Flight Data: {data}"),
    ])
    chain = prompt | llm
    try:
        resp = chain.invoke({"rules": rules_text, "data": json.dumps(case)})
        content = resp.content.strip()
        if "High Risk" in content:
            return "High Risk"
        return "Low Risk"
    except Exception:
        return "Error"


def main():
    llm = get_llm()
    if not llm:
        print("[ERROR] DEEPSEEK_API_KEY not set. Cannot reproduce Table 3.")
        print("  Set it via:  export DEEPSEEK_API_KEY='your_key'")
        sys.exit(1)

    dataset_path = current_dir / "data" / "ksem_large_dataset.json"
    if not dataset_path.exists():
        print("[ERROR] Dataset not found. Run generate_large_scale_dataset.py first.")
        sys.exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        all_cases = json.load(f)

    print(f"[INFO] Loaded {len(all_cases)} cases.")

    np.random.seed(42)

    results = {}

    for level_name, sigma in NOISE_LEVELS.items():
        print(f"\n{'='*60}")
        print(f"  Noise Level: {level_name} (sigma={sigma})")
        print(f"{'='*60}")

        gt_labels = []
        pred_rule, pred_llm, pred_skykg = [], [], []

        for i, case in enumerate(all_cases):
            gt_labels.append(case["ground_truth"])
            noisy_case = inject_noise(case, sigma)

            pred_rule.append(run_baseline_rule(noisy_case))
            pred_llm.append(run_baseline_llm(noisy_case, llm))
            pred_skykg.append(run_skykg(noisy_case, llm))

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(all_cases)} cases...", flush=True)

        valid_llm = [(g, p) for g, p in zip(gt_labels, pred_llm) if p != "Error"]
        valid_skykg = [(g, p) for g, p in zip(gt_labels, pred_skykg) if p != "Error"]

        acc_rule = accuracy_score(gt_labels, pred_rule)
        acc_llm = accuracy_score(
            [g for g, _ in valid_llm], [p for _, p in valid_llm]
        ) if valid_llm else 0.0
        acc_skykg = accuracy_score(
            [g for g, _ in valid_skykg], [p for _, p in valid_skykg]
        ) if valid_skykg else 0.0

        results[level_name] = {
            "Baseline-Rule": round(acc_rule, 3),
            "Baseline-LLM": round(acc_llm, 3),
            "SkyKG-Ours": round(acc_skykg, 3),
        }

        print(f"\n  Baseline-Rule: {acc_rule:.3f}")
        print(f"  Baseline-LLM:  {acc_llm:.3f}  ({len(valid_llm)}/{len(gt_labels)} valid)")
        print(f"  SkyKG-Ours:    {acc_skykg:.3f}  ({len(valid_skykg)}/{len(gt_labels)} valid)")

    print(f"\n\n{'='*70}")
    print("  Table 3: Robustness to Sensor Noise — Accuracy Under Varying Noise")
    print(f"{'='*70}")
    header = f"{'Method':<16} {'Mild (σ=0.05)':>14} {'Moderate (σ=0.15)':>18} {'Severe (σ=0.30)':>16}"
    print(header)
    print("-" * len(header))
    for method in ("Baseline-Rule", "Baseline-LLM", "SkyKG-Ours"):
        mild = results["Mild"][method]
        mod = results["Moderate"][method]
        sev = results["Severe"][method]
        print(f"{method:<16} {mild:>14.3f} {mod:>18.3f} {sev:>16.3f}")

    out_path = OUTPUT_DIR / "table3_robustness.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
