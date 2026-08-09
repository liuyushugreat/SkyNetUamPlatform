#!/usr/bin/env python3
"""Offline evaluation of label-isolated entity grounding on saved LLM responses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from skyrescue.entity_grounding import (
    anchors_equivalent,
    compile_grounded_candidate,
    ground_target,
    normalize_entity_text,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def evaluate_provider(
    provider: str,
    gold_rows: list[dict[str, Any]],
    response_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gold_by_id = {row["instruction_id"]: row for row in gold_rows}
    details: list[dict[str, Any]] = []
    counts = Counter()

    for response in response_rows:
        instruction_id = response["instruction_id"]
        gold = gold_by_id[instruction_id]
        counts["cases"] += 1
        parsed = response.get("parsed") if response.get("json_parse_success") else None
        parse_success = isinstance(parsed, dict)
        schema_valid = bool(response.get("schema_valid"))
        counts["parse_success"] += int(parse_success)
        counts["schema_valid"] += int(schema_valid)

        gold_anchor = ground_target(
            gold["target_zone"], gold["scenario_card"], gold["instruction_text"]
        )
        counts["gold_resolved"] += int(gold_anchor.resolved)

        predicted_target = str(parsed.get("target_zone", "")) if parse_success else ""
        predicted_anchor = ground_target(
            predicted_target, gold["scenario_card"], gold["instruction_text"]
        )
        counts["predicted_resolved"] += int(predicted_anchor.resolved)

        strict_match = (
            parse_success
            and normalize_entity_text(predicted_target)
            == normalize_entity_text(gold["target_zone"])
        )
        anchor_match = parse_success and anchors_equivalent(predicted_anchor, gold_anchor)
        counts["strict_match"] += int(strict_match)
        counts["anchor_match"] += int(anchor_match)

        gold_unresolved = not gold_anchor.resolved
        explicitly_unresolved = gold_anchor.reason == "explicit_unresolved"
        counts["gold_unresolved"] += int(gold_unresolved)
        counts["gold_explicitly_unresolved"] += int(explicitly_unresolved)
        counts["false_grounding"] += int(explicitly_unresolved and predicted_anchor.resolved)

        grounded_result = None
        executable = False
        failure = "SchemaInvalid"
        if parse_success and schema_valid:
            grounded_result = compile_grounded_candidate(
                parsed, gold["scenario_card"], gold["instruction_text"]
            )
            executable = grounded_result.compilation.executable
            failure = grounded_result.compilation.failure
            counts["grounding_gate"] += int(failure == "UngroundedEntity")

        expected_failure = gold["expected_failure"]
        gold_should_execute = expected_failure == "None"
        decision_correct = executable == gold_should_execute
        exact_failure = (
            decision_correct
            if gold_should_execute
            else failure == expected_failure
        )
        counts["decision_correct"] += int(decision_correct)
        counts["exact_outcome"] += int(exact_failure)
        counts["gold_valid"] += int(gold_should_execute)
        counts["gold_valid_executable"] += int(gold_should_execute and executable)

        details.append(
            {
                "provider": provider,
                "instruction_id": instruction_id,
                "scenario_group": gold["scenario_group"],
                "predicted_target": predicted_target,
                "gold_target": gold["target_zone"],
                "strict_target_match": bool(strict_match),
                "predicted_anchor": predicted_anchor.to_dict(),
                "gold_anchor": gold_anchor.to_dict(),
                "anchor_match": bool(anchor_match),
                "gold_expected_failure": expected_failure,
                "grounded_executable": executable,
                "grounded_failure": failure,
                "safe_compile_decision_correct": decision_correct,
                "exact_outcome_correct": exact_failure,
            }
        )

    total = counts["cases"]
    gold_valid = counts["gold_valid"]
    gold_explicitly_unresolved = counts["gold_explicitly_unresolved"]
    metrics = {
        "provider": provider,
        "cases": total,
        "direct_json_rate": ratio(counts["parse_success"], total),
        "schema_validation_rate": ratio(counts["schema_valid"], total),
        "strict_target_accuracy": ratio(counts["strict_match"], total),
        "predicted_entity_resolution_rate": ratio(counts["predicted_resolved"], total),
        "gold_entity_resolution_rate": ratio(counts["gold_resolved"], total),
        "anchored_target_accuracy": ratio(counts["anchor_match"], total),
        "false_grounding_rate_on_explicit_unresolved_gold": ratio(
            counts["false_grounding"], gold_explicitly_unresolved
        ),
        "grounding_gate_rate": ratio(counts["grounding_gate"], total),
        "safe_compile_decision_accuracy": ratio(counts["decision_correct"], total),
        "exact_compile_outcome_accuracy": ratio(counts["exact_outcome"], total),
        "grounded_executable_rate_on_gold_valid": ratio(counts["gold_valid_executable"], gold_valid),
        "gold_valid_cases": gold_valid,
        "ontology_abstention_cases": counts["gold_unresolved"],
        "gold_explicitly_unresolved_cases": gold_explicitly_unresolved,
        "false_grounding_cases": counts["false_grounding"],
    }
    return metrics, details


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_details_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "provider", "instruction_id", "scenario_group", "predicted_target", "gold_target",
        "strict_target_match", "predicted_anchor_ids", "gold_anchor_ids", "anchor_match",
        "predicted_grounding_reason", "gold_grounding_reason", "gold_expected_failure",
        "grounded_executable", "grounded_failure", "safe_compile_decision_correct",
        "exact_outcome_correct",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "provider": row["provider"],
                "instruction_id": row["instruction_id"],
                "scenario_group": row["scenario_group"],
                "predicted_target": row["predicted_target"],
                "gold_target": row["gold_target"],
                "strict_target_match": row["strict_target_match"],
                "predicted_anchor_ids": "|".join(row["predicted_anchor"]["anchor_ids"]),
                "gold_anchor_ids": "|".join(row["gold_anchor"]["anchor_ids"]),
                "anchor_match": row["anchor_match"],
                "predicted_grounding_reason": row["predicted_anchor"]["reason"],
                "gold_grounding_reason": row["gold_anchor"]["reason"],
                "gold_expected_failure": row["gold_expected_failure"],
                "grounded_executable": row["grounded_executable"],
                "grounded_failure": row["grounded_failure"],
                "safe_compile_decision_correct": row["safe_compile_decision_correct"],
                "exact_outcome_correct": row["exact_outcome_correct"],
            })


def write_markdown(path: Path, metrics: list[dict[str, Any]]) -> None:
    lines = [
        "# SkyRescue 无标签泄漏地点实体锚定评估",
        "",
        "本次评估复用已保存的 200 份模型响应，没有再次调用 API。在线锚定仅读取场景卡、指挥指令和模型预测目标；标准答案只在离线评估阶段被独立锚定后用于比较。",
        "",
        "| 模型 | 直接 JSON | Schema 校验 | 严格地点文本 | 锚定地点 | 安全编译决策 | 精确编译结果 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            f"| {row['provider']} | {row['direct_json_rate']:.2%} | "
            f"{row['schema_validation_rate']:.2%} | {row['strict_target_accuracy']:.2%} | "
            f"{row['anchored_target_accuracy']:.2%} | {row['safe_compile_decision_accuracy']:.2%} | "
            f"{row['exact_compile_outcome_accuracy']:.2%} |"
        )
    lines.extend([
        "",
        "## 安全诊断",
        "",
        "| 模型 | 预测实体解析率 | 锚定阻断率 | 标准有效案例可执行率 | 未解析标准上的错误强制锚定 |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in metrics:
        lines.append(
            f"| {row['provider']} | {row['predicted_entity_resolution_rate']:.2%} | "
            f"{row['grounding_gate_rate']:.2%} | {row['grounded_executable_rate_on_gold_valid']:.2%} | "
            f"{row['false_grounding_rate_on_explicit_unresolved_gold']:.2%} "
            f"({row['false_grounding_cases']}/{row['gold_explicitly_unresolved_cases']}) |"
        )
    lines.extend([
        "",
        "注：“锚定地点”比较独立解析后的冻结本体 ID，不等同于字面完全一致。无法唯一锚定的泛指地点会被 `UngroundedEntity` 阻断，不会进入执行。",
        "",
        "研究边界：该本体在检查过既有模型误差后开发，因此本表是事后开发集证据。投稿时应冻结本体及阈值，再用未参与开发的新指令集做确证性评估。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--response-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--providers", nargs="+", default=["deepseek", "qwen"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gold_rows = read_jsonl(args.input)
    all_metrics = []
    all_details = []
    response_hashes = {}
    for provider in args.providers:
        response_path = args.response_dir / f"raw_{provider}.jsonl"
        response_rows = read_jsonl(response_path)
        metrics, details = evaluate_provider(provider, gold_rows, response_rows)
        all_metrics.append(metrics)
        all_details.extend(details)
        response_hashes[provider] = sha256(response_path)

    write_summary_csv(args.output_dir / "entity_grounding_summary.csv", all_metrics)
    write_details_csv(args.output_dir / "entity_grounding_details.csv", all_details)
    write_jsonl(args.output_dir / "entity_grounding_details.jsonl", all_details)
    (args.output_dir / "entity_grounding_summary.json").write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(args.output_dir / "PAPER_RESULTS_zh.md", all_metrics)
    manifest = {
        "experiment": "SkyRescue-LabelIsolated-EntityGrounding-v1.0.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "response_dir": str(args.response_dir.resolve()),
        "response_sha256": response_hashes,
        "providers": args.providers,
        "saved_responses_reused": True,
        "api_calls": 0,
        "online_grounding_inputs": ["scenario_card", "instruction_text", "predicted_target_zone"],
        "gold_target_usage": "offline_independent_grounding_and_comparison_only",
        "analysis_status": "post_hoc_development_evaluation",
        "confirmatory_requirement": "freeze_ontology_then_evaluate_on_unseen_instructions",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(all_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
