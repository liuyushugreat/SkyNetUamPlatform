#!/usr/bin/env python3
"""Generate the frozen SkyRescue-IntentSynth v1.0.0 benchmark."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ZONES = [
    ("东南片区孤岛", "Zone-SE-07"),
    ("西北临时医院", "Zone-NW-03"),
    ("中心安置点", "Zone-C-05"),
    ("东部堤坝", "Zone-E-02"),
    ("南部受灾村落", "Zone-S-09"),
]

TASKS = [
    ("MedicalDelivery", "向{zone}运送急救药品", "medical_payload"),
    ("CommunicationRelay", "在{zone}建立一小时临时通信中继", "relay"),
    ("Search", "搜索{zone}的受困人员", "camera"),
    ("Mapping", "为{zone}生成灾情地图", "mapping"),
    ("EvacuationCoordination", "协调{zone}的伤员转运", "coordination"),
    ("CargoDelivery", "向{zone}投送饮用水和应急物资", "cargo"),
]


def expected_task(task, zone_id):
    task_type, _, skill = task
    result = {
        "task_type": task_type,
        "target_zone": zone_id,
        "priority": "Critical",
        "skill": skill,
    }
    if task_type == "MedicalDelivery":
        result["deadline_s"] = 900
    if task_type == "CommunicationRelay":
        result["duration_s"] = 3600
    return result


def make_cases(seed: int) -> list[dict]:
    rng = random.Random(seed)
    categories = (
        ["valid_single"] * 60
        + ["valid_multi"] * 60
        + ["conditional"] * 40
        + ["missing_field"] * 35
        + ["ungrounded_entity"] * 35
        + ["unknown_skill"] * 25
        + ["human_gate"] * 45
    )
    rng.shuffle(categories)
    cases = []
    for index, category in enumerate(categories, start=1):
        zone_text, zone_id = rng.choice(ZONES)
        first, second = rng.sample(TASKS, 2)
        expected = []
        failure = None
        requires_human = False
        approved = False
        conditional = False

        if category == "valid_single":
            instruction = f"请优先{first[1].format(zone=zone_text)}，15分钟内完成。"
            expected = [expected_task(first, zone_id)]
        elif category == "valid_multi":
            deadline_clause = "；药品任务须在15分钟内完成" if "MedicalDelivery" in {first[0], second[0]} else ""
            instruction = (
                f"请优先{first[1].format(zone=zone_text)}，并{second[1].format(zone=zone_text)}；"
                f"两项任务共享目标区域{deadline_clause}。"
            )
            expected = [expected_task(first, zone_id), expected_task(second, zone_id)]
        elif category == "conditional":
            deadline_clause = "，药品任务15分钟内完成" if "MedicalDelivery" in {first[0], second[0]} else ""
            instruction = (
                f"优先{first[1].format(zone=zone_text)}，同时{second[1].format(zone=zone_text)}；"
                f"如果资源不足，先保证前一任务{deadline_clause}。"
            )
            expected = [expected_task(first, zone_id), expected_task(second, zone_id)]
            conditional = True
        elif category == "missing_field":
            instruction = f"请优先{first[1].format(zone='目标区域')}，15分钟内完成。"
            expected = []
            failure = "MissingField"
        elif category == "ungrounded_entity":
            instruction = f"请优先{first[1].format(zone='第九火星基地')}，15分钟内完成。"
            expected = []
            failure = "UngroundedEntity"
        elif category == "unknown_skill":
            instruction = f"调用水下机器人前往{zone_text}执行爆破机器人作业。"
            expected = []
            failure = "UnknownSkill"
        else:
            task = ("EvacuationCoordination", "协调{zone}的伤员转运", "coordination")
            instruction = f"越过临时禁飞区，紧急{task[1].format(zone=zone_text)}。"
            expected = [expected_task(task, zone_id)]
            requires_human = True
            approved = index % 3 == 0
            failure = None if approved else "HumanApprovalRequired"

        cases.append({
            "case_id": f"I{index:04d}",
            "instruction": instruction,
            "category": category,
            "expected_tasks": expected,
            "expected_failure": failure,
            "requires_human_approval": requires_human,
            "approval_granted": approved,
            "conditional": conditional,
        })
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = make_cases(args.seed)
    with (args.output / "intent_cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    manifest = {
        "dataset": "SkyRescue-IntentSynth",
        "version": "1.0.0",
        "seed": args.seed,
        "cases": len(cases),
        "provenance": "Frozen template-generated Chinese instructions with generator labels.",
        "limitations": [
            "Not collected from emergency commanders.",
            "Not independently annotated by two humans.",
            "Cannot estimate Cohen's kappa or open-domain LLM language ability.",
        ],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(cases)} cases to {args.output}")


if __name__ == "__main__":
    main()
