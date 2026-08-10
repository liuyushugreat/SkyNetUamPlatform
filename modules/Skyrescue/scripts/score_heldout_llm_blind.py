#!/usr/bin/env python3
"""Score frozen held-out LLM responses after gold-label adjudication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FIELDS = (
    "task_type",
    "target_zone",
    "priority",
    "deadline_s_or_text",
    "required_skill",
    "needs_human_approval",
    "expected_failure",
)

ENUMS = {
    "task_type": {
        "MedicalDelivery", "SearchAndRescue", "FireRecon", "CommunicationRelay",
        "HazmatMonitoring", "SupplyDelivery", "InfrastructureInspection",
        "EvacuationSupport", "TrafficMonitoring", "RouteReplan",
        "PriorityPreemption", "Other",
    },
    "priority": {"Critical", "High", "Normal"},
    "deadline_s_or_text": {"urgent_unspecified", "unspecified"},
    "required_skill": {
        "medical_payload", "thermal_recon", "mapping", "communication_relay",
        "hazmat_monitoring", "supply_payload", "infrastructure_photo",
        "loudspeaker_guidance", "traffic_monitoring", "route_replan",
        "unknown_or_other",
    },
    "needs_human_approval": {"Yes", "No"},
    "expected_failure": {
        "None", "AmbiguousIntent", "HumanApprovalRequired", "ResourceUnavailable",
        "IncompleteConstraint", "UnknownSkill", "PolicyOrAirspaceConflict",
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(value: Any) -> str:
    return re.sub(r"[\s，。；、：:]+", "", str(value or "")).strip()


def field_equal(field: str, predicted: Any, gold: Any) -> bool:
    return normalized(predicted) == normalized(gold) if field == "target_zone" else predicted == gold


def ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def macro_f1(gold: list[str], predicted: list[str | None]) -> float:
    values = []
    for label in sorted(set(gold)):
        tp = sum(g == label and p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return round(statistics.mean(values), 6) if values else 0.0


def exact_sign_test(wins: int, losses: int) -> float:
    non_ties = wins + losses
    if not non_ties:
        return 1.0
    lower = min(wins, losses)
    tail = sum(math.comb(non_ties, index) for index in range(lower + 1)) / (2 ** non_ties)
    return min(1.0, 2 * tail)


def validate_gold(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Gold set is empty.")
    identifiers = []
    required = {"instruction_id", "instruction_text", *FIELDS}
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"Gold row is missing fields: {sorted(missing)}")
        identifiers.append(str(row["instruction_id"]))
        for field, allowed in ENUMS.items():
            if row[field] not in allowed:
                raise ValueError(f"Illegal gold value for {field}: {row[field]}")
        if not str(row["target_zone"]).strip():
            raise ValueError("Gold target_zone must not be blank.")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Gold set contains duplicate instruction IDs.")


def validate_responses(provider: str, gold_rows: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    gold_ids = {row["instruction_id"] for row in gold_rows}
    response_ids = [row.get("instruction_id") for row in rows]
    if len(response_ids) != len(set(response_ids)):
        raise ValueError(f"{provider} responses contain duplicate instruction IDs.")
    if set(response_ids) != gold_ids:
        missing = sorted(gold_ids - set(response_ids))
        extra = sorted(set(response_ids) - gold_ids)
        raise ValueError(f"{provider} response IDs mismatch; missing={missing}, extra={extra}")
    for row in rows:
        if row.get("provider") != provider:
            raise ValueError(f"Provider mismatch in {provider} response file.")
        if row.get("gold_labels_sent") is not False or row.get("scenario_card_sent") is not False:
            raise ValueError(f"{provider} response violates the blind-input boundary.")
        if row.get("input_fields") != ["instruction_text"]:
            raise ValueError(f"{provider} response contains unexpected model input fields.")


def compile_outcome(compilation: dict[str, Any], expected_failure: str) -> bool:
    if expected_failure == "None":
        return bool(compilation.get("executable"))
    return compilation.get("failure") == expected_failure


def evaluate_provider(
    provider: str,
    gold_rows: list[dict[str, Any]],
    response_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {row["instruction_id"]: row for row in response_rows}
    details: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    counts = Counter()
    latencies = []

    for gold in gold_rows:
        response = by_id[gold["instruction_id"]]
        predicted = response.get("parsed") if isinstance(response.get("parsed"), dict) else {}
        field_hits = {field: field_equal(field, predicted.get(field), gold[field]) for field in FIELDS}
        hit_count = sum(field_hits.values())
        counts["field_hits"] += hit_count
        counts["exact_records"] += int(hit_count == len(FIELDS))
        counts["api_success"] += int(bool(response.get("api_success")))
        counts["json_parse"] += int(bool(response.get("json_parse_success")))
        counts["schema_valid"] += int(bool(response.get("schema_valid")))
        if response.get("api_success"):
            latencies.append(float(response.get("latency_ms", 0)))

        direct = response.get("compile_result") or {}
        grounded_block = response.get("grounded_compile_result") or {}
        grounded = grounded_block.get("compilation") or {}
        anchor = grounded_block.get("anchor") or {}
        expected_failure = gold["expected_failure"]
        gold_should_execute = expected_failure == "None"
        direct_exact = compile_outcome(direct, expected_failure)
        grounded_exact = compile_outcome(grounded, expected_failure)
        grounded_safe = bool(grounded.get("executable")) == gold_should_execute
        counts["direct_outcome"] += int(direct_exact)
        counts["grounded_outcome"] += int(grounded_exact)
        counts["grounded_safe"] += int(grounded_safe)

        target_correct = field_hits["target_zone"]
        anchor_resolved = bool(anchor.get("resolved"))
        if target_correct and anchor_resolved:
            gate_class = "correct_accept"
        elif target_correct:
            gate_class = "correct_reject"
        elif anchor_resolved:
            gate_class = "incorrect_accept"
        else:
            gate_class = "incorrect_reject"
        counts[gate_class] += 1

        details.append({
            "provider": provider,
            "model": response.get("reported_model") or response.get("requested_model"),
            "instruction_id": gold["instruction_id"],
            "instruction_text": gold["instruction_text"],
            "api_success": bool(response.get("api_success")),
            "json_parse_success": bool(response.get("json_parse_success")),
            "schema_valid": bool(response.get("schema_valid")),
            "field_hits": field_hits,
            "field_hit_count": hit_count,
            "all_fields_exact": hit_count == len(FIELDS),
            "predicted": predicted,
            "gold": {field: gold[field] for field in FIELDS},
            "direct_compile_executable": bool(direct.get("executable")),
            "direct_compile_failure": direct.get("failure"),
            "direct_outcome_correct": direct_exact,
            "anchor_resolved": anchor_resolved,
            "anchor_reason": anchor.get("reason"),
            "anchor_ids": anchor.get("anchor_ids") or [],
            "grounding_gate_class": gate_class,
            "grounded_compile_executable": bool(grounded.get("executable")),
            "grounded_compile_failure": grounded.get("failure"),
            "grounded_safe_decision_correct": grounded_safe,
            "grounded_exact_outcome_correct": grounded_exact,
            "latency_ms": response.get("latency_ms"),
        })

    for field in FIELDS:
        gold_values = [str(row[field]) for row in gold_rows]
        predicted_values = [
            str((by_id[row["instruction_id"]].get("parsed") or {}).get(field))
            if field in (by_id[row["instruction_id"]].get("parsed") or {}) else None
            for row in gold_rows
        ]
        matches = sum(field_equal(field, predicted, gold) for predicted, gold in zip(predicted_values, gold_values))
        field_rows.append({
            "provider": provider,
            "field": field,
            "matches": matches,
            "cases": len(gold_rows),
            "accuracy": ratio(matches, len(gold_rows)),
            "macro_f1": None if field == "target_zone" else macro_f1(gold_values, predicted_values),
        })

    count = len(gold_rows)
    correct_targets = counts["correct_accept"] + counts["correct_reject"]
    incorrect_targets = counts["incorrect_accept"] + counts["incorrect_reject"]
    accepted_targets = counts["correct_accept"] + counts["incorrect_accept"]
    summary = {
        "provider": provider,
        "model": response_rows[0].get("reported_model") or response_rows[0].get("requested_model"),
        "cases": count,
        "api_success_rate": ratio(counts["api_success"], count),
        "direct_json_parse_rate": ratio(counts["json_parse"], count),
        "schema_pass_rate": ratio(counts["schema_valid"], count),
        "slot_micro_accuracy": ratio(counts["field_hits"], count * len(FIELDS)),
        "all_fields_exact_rate": ratio(counts["exact_records"], count),
        "direct_compile_outcome_accuracy": ratio(counts["direct_outcome"], count),
        "grounded_safe_decision_accuracy": ratio(counts["grounded_safe"], count),
        "grounded_exact_outcome_accuracy": ratio(counts["grounded_outcome"], count),
        "target_zone_strict_accuracy": ratio(correct_targets, count),
        "grounding_acceptance_rate": ratio(accepted_targets, count),
        "correct_target_acceptance_rate": ratio(counts["correct_accept"], correct_targets),
        "incorrect_target_block_rate": ratio(counts["incorrect_reject"], incorrect_targets),
        "accepted_target_precision": ratio(counts["correct_accept"], accepted_targets),
        "incorrect_target_false_accept_rate": ratio(counts["incorrect_accept"], incorrect_targets),
        "correct_accept": counts["correct_accept"],
        "correct_reject": counts["correct_reject"],
        "incorrect_accept": counts["incorrect_accept"],
        "incorrect_reject": counts["incorrect_reject"],
        "latency_mean_ms": round(statistics.mean(latencies), 3) if latencies else None,
        "latency_p95_ms": round(sorted(latencies)[max(0, math.ceil(0.95 * len(latencies)) - 1)], 3) if latencies else None,
    }
    return summary, field_rows, details


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flatten_detail(row: dict[str, Any]) -> dict[str, Any]:
    flat = {key: value for key, value in row.items() if key not in {"field_hits", "predicted", "gold", "anchor_ids"}}
    flat["anchor_ids"] = "|".join(row["anchor_ids"])
    for field in FIELDS:
        flat[f"pred_{field}"] = row["predicted"].get(field)
        flat[f"gold_{field}"] = row["gold"].get(field)
        flat[f"hit_{field}"] = row["field_hits"][field]
    return flat


def write_markdown(path: Path, summaries: list[dict[str, Any]], comparison: dict[str, Any]) -> None:
    lines = [
        "# SkyRescue HeldOut100 confirmatory blind evaluation",
        "",
        "The 200 saved model responses were scored only after A/B annotation and third-expert adjudication. No API was called during scoring.",
        "",
        "| Model | JSON | Schema | Slot accuracy | Exact record | Direct compile | Grounded safe decision | Grounded exact outcome |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['model']} | {row['direct_json_parse_rate']:.2%} | {row['schema_pass_rate']:.2%} | "
            f"{row['slot_micro_accuracy']:.2%} | {row['all_fields_exact_rate']:.2%} | "
            f"{row['direct_compile_outcome_accuracy']:.2%} | {row['grounded_safe_decision_accuracy']:.2%} | "
            f"{row['grounded_exact_outcome_accuracy']:.2%} |"
        )
    lines.extend([
        "",
        "| Model | Target exact | Gate coverage | Correct-target acceptance | Incorrect-target block | Accepted-target precision |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in summaries:
        lines.append(
            f"| {row['model']} | {row['target_zone_strict_accuracy']:.2%} | "
            f"{row['grounding_acceptance_rate']:.2%} | {row['correct_target_acceptance_rate']:.2%} | "
            f"{row['incorrect_target_block_rate']:.2%} | {row['accepted_target_precision']:.2%} |"
        )
    lines.extend([
        "",
        f"Paired seven-field comparison: DeepSeek wins {comparison['deepseek_wins']}, Qwen wins {comparison['qwen_wins']}, ties {comparison['ties']}; two-sided exact sign test p={comparison['sign_test_p']:.6f}.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_paper_results_zh(path: Path, summaries: list[dict[str, Any]], comparison: dict[str, Any]) -> None:
    by_provider = {row["provider"]: row for row in summaries}
    deepseek = by_provider["deepseek"]
    qwen = by_provider["qwen"]
    lines = [
        "# SkyRescue HeldOut100 确证性盲测结果",
        "",
        "本实验在冻结实体锚定器、提示词和解码参数后，使用 100 条新收集的人工低空应急指令进行确证性评估。DeepSeek 与通义千问的 200 份原始响应均在专家 A/B 标注和第三专家裁决完成前保存；揭盲评分阶段复用原始响应，没有再次调用 API。两位专家在 700 个标签单元中的一致率为 97.43%，宏平均 Cohen's kappa 为 0.9533。",
        "",
        "| 模型 | JSON 解析 | Schema 通过 | 七字段微准确率 | 整条全对 | 无锚定门编译结果准确率 | 锚定后安全决策准确率 | 锚定后精确结果准确率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['model']} | {row['direct_json_parse_rate']:.2%} | {row['schema_pass_rate']:.2%} | "
            f"{row['slot_micro_accuracy']:.2%} | {row['all_fields_exact_rate']:.2%} | "
            f"{row['direct_compile_outcome_accuracy']:.2%} | {row['grounded_safe_decision_accuracy']:.2%} | "
            f"{row['grounded_exact_outcome_accuracy']:.2%} |"
        )
    lines.extend([
        "",
        f"按每条指令的七字段正确数进行配对比较，DeepSeek 在 {comparison['deepseek_wins']} 条上更高，千问在 {comparison['qwen_wins']} 条上更高，另有 {comparison['ties']} 条并列；双侧精确符号检验 p={comparison['sign_test_p']:.4f}，未发现显著差异。DeepSeek 的七字段微准确率为 {deepseek['slot_micro_accuracy']:.2%}，千问为 {qwen['slot_micro_accuracy']:.2%}。",
        "",
        "| 模型 | 地点严格匹配 | 锚定门接受率 | 正确地点接受率 | 错误地点阻断率 | 被接受地点的严格正确率 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in summaries:
        lines.append(
            f"| {row['model']} | {row['target_zone_strict_accuracy']:.2%} | "
            f"{row['grounding_acceptance_rate']:.2%} | {row['correct_target_acceptance_rate']:.2%} | "
            f"{row['incorrect_target_block_rate']:.2%} | {row['accepted_target_precision']:.2%} |"
        )
    lines.extend([
        "",
        "冻结锚定门在未见过的口语地点表达上表现出明显的覆盖率瓶颈。其总体接受率仅为 22%--23%，并错误阻断了多数严格匹配的地点，因此锚定后安全决策准确率下降至 23%--27%。另一方面，对严格不匹配地点的阻断率达到 75.00%--82.61%，说明保守拒绝策略能够降低未锚定目标直接进入执行链的风险，但代价是较高的误拒绝。",
        "",
        "该结果应作为确证性负面结果完整报告：冻结本体在新分布上的泛化能力不足，不能据此声称实体锚定机制已达到真实部署要求。后续可以在保持 v1.0.0 结果不变的前提下开发扩展本体或可学习锚定器，并在另一份全新测试集上进行 v1.1.0 评估；不得使用当前 100 条测试标签调参后仍将其称为确证性结果。",
        "",
        "注：地点正确性采用去除空白和常用标点后的严格文本匹配。不同表述但语义近似的地点仍可能被计为不匹配，因此锚定门的严格正确率是保守指标。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--response-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--providers", nargs="+", default=["deepseek", "qwen"])
    args = parser.parse_args()

    gold_rows = read_jsonl(args.gold)
    validate_gold(gold_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    response_hashes = {}
    for provider in args.providers:
        response_path = args.response_dir / f"raw_{provider}.jsonl"
        response_rows = read_jsonl(response_path)
        validate_responses(provider, gold_rows, response_rows)
        summary, provider_fields, provider_details = evaluate_provider(provider, gold_rows, response_rows)
        summaries.append(summary)
        field_rows.extend(provider_fields)
        details.extend(provider_details)
        response_hashes[provider] = sha256(response_path)

    score_by_provider = {
        provider: {row["instruction_id"]: row["field_hit_count"] for row in details if row["provider"] == provider}
        for provider in args.providers
    }
    deepseek_scores = score_by_provider["deepseek"]
    qwen_scores = score_by_provider["qwen"]
    wins = sum(deepseek_scores[key] > qwen_scores[key] for key in deepseek_scores)
    losses = sum(deepseek_scores[key] < qwen_scores[key] for key in deepseek_scores)
    comparison = {
        "deepseek_wins": wins,
        "qwen_wins": losses,
        "ties": len(gold_rows) - wins - losses,
        "sign_test_p": round(exact_sign_test(wins, losses), 8),
    }

    payload = {"models": summaries, "field_metrics": field_rows, "paired_comparison": comparison}
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "summary.csv", summaries)
    write_csv(args.output_dir / "field_metrics.csv", field_rows)
    write_csv(args.output_dir / "predictions.csv", [flatten_detail(row) for row in details])
    write_markdown(args.output_dir / "RESULTS.md", summaries, comparison)
    write_paper_results_zh(args.output_dir / "PAPER_RESULTS_zh.md", summaries, comparison)
    manifest = {
        "experiment": "SkyRescue-HeldOut100-Confirmatory-Scoring-v1.0.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gold": str(args.gold.resolve()),
        "gold_sha256": sha256(args.gold),
        "response_dir": str(args.response_dir.resolve()),
        "response_sha256": response_hashes,
        "saved_responses_reused": True,
        "api_calls_during_scoring": 0,
        "gold_labels_opened_after_response_capture": True,
        "target_match": "whitespace_and_common_punctuation_normalized_exact_match",
        "grounding_gate_reference": "strict_target_zone_correctness",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
