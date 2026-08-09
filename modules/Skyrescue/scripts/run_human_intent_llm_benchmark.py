#!/usr/bin/env python3
"""Run frozen DeepSeek/Qwen intent extraction on the human gold set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import ssl
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from skyrescue.workflow import compile_generated_candidate


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

SYSTEM_PROMPT = """你是低空应急任务的候选任务生成器。请依据场景背景和指挥指令抽取一个结构化任务。
只输出一个 JSON 对象，不得输出 Markdown、代码围栏、解释或额外字段。必须恰好包含以下七个字符串字段：
task_type, target_zone, priority, deadline_s_or_text, required_skill, needs_human_approval, expected_failure。

允许值：
task_type: MedicalDelivery | SearchAndRescue | FireRecon | CommunicationRelay | HazmatMonitoring | SupplyDelivery | InfrastructureInspection | EvacuationSupport | TrafficMonitoring | RouteReplan | PriorityPreemption | Other
priority: Critical | High | Normal
deadline_s_or_text: urgent_unspecified | unspecified
required_skill: medical_payload | thermal_recon | mapping | communication_relay | hazmat_monitoring | supply_payload | infrastructure_photo | loudspeaker_guidance | traffic_monitoring | route_replan | unknown_or_other
needs_human_approval: Yes | No
expected_failure: None | AmbiguousIntent | HumanApprovalRequired | ResourceUnavailable | IncompleteConstraint | UnknownSkill | PolicyOrAirspaceConflict

判定原则：target_zone 使用指令中最具体、可执行的中文地点短语；“马上、尽快、第一时间、越快越好”等记为 urgent_unspecified；没有明确紧迫表达记为 unspecified。涉及跨禁飞区、敏感载荷、人工确认后才能执行、暂停/恢复已提交任务等需要人工授权的操作时，needs_human_approval 为 Yes。若信息足以形成候选任务，expected_failure 为 None；否则选择最主要的结构化失败类型。"""

USER_TEMPLATE = "场景背景：{scenario_card}\n指挥指令：{instruction_text}"

PROVIDERS = {
    "deepseek": {
        "env": "DEEPSEEK_API_KEY",
        "model": "deepseek-v4-flash",
        "url": "https://api.deepseek.com/chat/completions",
    },
    "qwen": {
        "env": "DASHSCOPE_API_KEY",
        "model": "qwen3-30b-a3b-instruct-2507",
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    },
}

SYSTEM_CA_FILE = Path("/etc/ssl/cert.pem")
SSL_CONTEXT = ssl.create_default_context(
    cafile=str(SYSTEM_CA_FILE) if SYSTEM_CA_FILE.exists() else None
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_secret(name: str, key_file: Path) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    text = key_file.read_text(encoding="utf-8")
    line = next((line for line in text.splitlines() if name in line), "")
    if not line:
        raise RuntimeError(f"{name} not found in environment or key file")
    tail = line.split(name, 1)[1]
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]{15,}", tail)
    if not tokens:
        raise RuntimeError(f"Could not parse {name} from key file")
    return tokens[-1]


def schema_errors(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["root_not_object"]
    errors = []
    keys = set(value)
    expected = set(FIELDS)
    if keys != expected:
        if expected - keys:
            errors.append("missing:" + ",".join(sorted(expected - keys)))
        if keys - expected:
            errors.append("extra:" + ",".join(sorted(keys - expected)))
    for field in FIELDS:
        if field not in value:
            continue
        if not isinstance(value[field], str):
            errors.append(f"type:{field}")
            continue
        if field == "target_zone":
            if not value[field].strip():
                errors.append("empty:target_zone")
        elif value[field] not in ENUMS[field]:
            errors.append(f"enum:{field}")
    return errors


def strict_json(content: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        return None, f"{exc.msg}@{exc.pos}"
    if not isinstance(value, dict):
        return None, "root_not_object"
    return value, None


def post_json(url: str, key: str, payload: dict[str, Any], timeout: int = 240) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout, context=SSL_CONTEXT) as response:
        return json.loads(response.read().decode("utf-8"))


def call_one(provider: str, case: dict[str, Any], key: str) -> dict[str, Any]:
    config = PROVIDERS[provider]
    user_prompt = USER_TEMPLATE.format(
        scenario_card=case["scenario_card"],
        instruction_text=case["instruction_text"],
    )
    payload = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 512,
        "stream": False,
    }
    if provider == "deepseek":
        payload["thinking"] = {"type": "disabled"}
    started = time.perf_counter()
    error = None
    response = None
    for attempt in range(1, 5):
        try:
            response = post_json(config["url"], key, payload)
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            error = f"HTTP {exc.code}: {body}"
            if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt == 4:
                break
        except Exception as exc:  # network and response decoding failures
            error = f"{type(exc).__name__}: {exc}"
            if attempt == 4:
                break
        time.sleep(2 ** (attempt - 1))

    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    base = {
        "provider": provider,
        "requested_model": config["model"],
        "instruction_id": case["instruction_id"],
        "latency_ms": latency_ms,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if response is None:
        return {**base, "api_success": False, "error": error}

    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        return {**base, "api_success": False, "error": f"malformed_response:{exc}"}

    parsed, parse_error = strict_json(content)
    errors = schema_errors(parsed) if parsed is not None else ["json_parse_failed"]
    compiled = compile_generated_candidate(parsed).to_dict() if parsed is not None and not errors else None
    return {
        **base,
        "api_success": True,
        "reported_model": response.get("model"),
        "usage": response.get("usage", {}),
        "content": content,
        "json_parse_success": parsed is not None,
        "json_parse_error": parse_error,
        "parsed": parsed,
        "schema_valid": not errors,
        "schema_errors": errors,
        "compile_result": compiled,
    }


def normalized(value: Any) -> str:
    return re.sub(r"[\s，。；、：:]+", "", str(value or "")).strip()


def field_equal(field: str, predicted: Any, gold: Any) -> bool:
    if field == "target_zone":
        return normalized(predicted) == normalized(gold)
    return predicted == gold


def macro_f1(gold: list[str], predicted: list[str | None]) -> float:
    labels = sorted(set(gold))
    values = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return statistics.mean(values) if values else 0.0


def evaluate(provider: str, cases: list[dict[str, Any]], records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {row["instruction_id"]: row for row in records if row["provider"] == provider}
    details = []
    field_rows = []
    api_successes = parses = schema_passes = exact_records = slot_hits = 0
    compile_outcome_hits = valid_cases = executable_valid = invalid_cases = failure_hits = 0
    latencies = []

    for case in cases:
        result = by_id.get(case["instruction_id"], {})
        predicted = result.get("parsed") or {}
        api_successes += int(bool(result.get("api_success")))
        parses += int(bool(result.get("json_parse_success")))
        schema_passes += int(bool(result.get("schema_valid")))
        if result.get("api_success"):
            latencies.append(float(result.get("latency_ms", 0)))
        field_hits = {field: field_equal(field, predicted.get(field), case[field]) for field in FIELDS}
        slot_hits += sum(field_hits.values())
        exact = all(field_hits.values())
        exact_records += int(exact)

        compile_result = result.get("compile_result") or {}
        gold_failure = case["expected_failure"]
        predicted_failure = compile_result.get("failure")
        if gold_failure == "None":
            valid_cases += 1
            executable_valid += int(bool(compile_result.get("executable")))
            compile_outcome_hits += int(bool(compile_result.get("executable")))
        else:
            invalid_cases += 1
            hit = predicted_failure == gold_failure
            failure_hits += int(hit)
            compile_outcome_hits += int(hit)

        details.append({
            "provider": provider,
            "model": result.get("reported_model") or result.get("requested_model", PROVIDERS[provider]["model"]),
            "instruction_id": case["instruction_id"],
            "scenario_group": case.get("scenario_group"),
            "instruction_text": case["instruction_text"],
            "api_success": bool(result.get("api_success")),
            "json_parse_success": bool(result.get("json_parse_success")),
            "schema_valid": bool(result.get("schema_valid")),
            "compile_executable": bool(compile_result.get("executable")),
            "compile_failure": predicted_failure,
            "all_fields_exact": exact,
            "field_hits": field_hits,
            "predicted": predicted,
            "gold": {field: case[field] for field in FIELDS},
            "latency_ms": result.get("latency_ms"),
            "error": result.get("error") or result.get("json_parse_error") or ";".join(result.get("schema_errors", [])),
        })

    for field in FIELDS:
        gold_values = [str(case[field]) for case in cases]
        predicted_values = [
            (str((by_id.get(case["instruction_id"], {}).get("parsed") or {}).get(field))
             if field in (by_id.get(case["instruction_id"], {}).get("parsed") or {}) else None)
            for case in cases
        ]
        accuracy = sum(field_equal(field, p, g) for p, g in zip(predicted_values, gold_values)) / len(cases)
        field_rows.append({
            "provider": provider,
            "field": field,
            "accuracy": round(accuracy, 4),
            "macro_f1": None if field == "target_zone" else round(macro_f1(gold_values, predicted_values), 4),
        })

    count = len(cases)
    summary = {
        "provider": provider,
        "requested_model": PROVIDERS[provider]["model"],
        "cases": count,
        "api_success_rate": round(api_successes / count, 4),
        "direct_json_parse_rate": round(parses / count, 4),
        "schema_pass_rate": round(schema_passes / count, 4),
        "all_fields_exact_rate": round(exact_records / count, 4),
        "slot_micro_accuracy": round(slot_hits / (count * len(FIELDS)), 4),
        "field_macro_accuracy": round(statistics.mean(row["accuracy"] for row in field_rows), 4),
        "compiler_outcome_accuracy": round(compile_outcome_hits / count, 4),
        "executable_workflow_rate_on_gold_valid": round(executable_valid / valid_cases, 4) if valid_cases else None,
        "structured_failure_accuracy": round(failure_hits / invalid_cases, 4) if invalid_cases else None,
        "latency_mean_ms": round(statistics.mean(latencies), 3) if latencies else None,
        "latency_p95_ms": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 3) if latencies else None,
    }
    return summary, field_rows, details


def write_outputs(output_dir: Path, cases: list[dict[str, Any]], providers: list[str]) -> None:
    records = []
    for provider in providers:
        path = output_dir / f"raw_{provider}.jsonl"
        if path.exists():
            records.extend(read_jsonl(path))

    summaries = []
    fields = []
    details = []
    for provider in providers:
        summary, field_rows, detail_rows = evaluate(provider, cases, records)
        summaries.append(summary)
        fields.extend(field_rows)
        details.extend(detail_rows)

    (output_dir / "summary.json").write_text(
        json.dumps({"models": summaries, "field_metrics": fields}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with (output_dir / "field_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields[0]))
        writer.writeheader()
        writer.writerows(fields)

    detail_headers = [
        "provider", "model", "instruction_id", "scenario_group", "instruction_text",
        "api_success", "json_parse_success", "schema_valid", "compile_executable",
        "compile_failure", "all_fields_exact", "latency_ms", "error",
    ] + [f"pred_{field}" for field in FIELDS] + [f"gold_{field}" for field in FIELDS]
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_headers)
        writer.writeheader()
        for row in details:
            flat = {key: row.get(key) for key in detail_headers if not key.startswith(("pred_", "gold_"))}
            flat.update({f"pred_{field}": row["predicted"].get(field) for field in FIELDS})
            flat.update({f"gold_{field}": row["gold"].get(field) for field in FIELDS})
            writer.writerow(flat)

    md = [
        "# SkyRescue human-instruction LLM benchmark",
        "",
        "Fixed parameters: temperature=0, top_p=1, max_tokens=512; one response per model and instruction.",
        "The same raw response is reused for direct JSON parsing, schema validation, and full SkyRescue compilation.",
        "",
        "| Model | API success | Direct JSON | Schema pass | Slot accuracy | Exact record | Compiler outcome |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        md.append(
            f"| {row['requested_model']} | {row['api_success_rate']:.4f} | "
            f"{row['direct_json_parse_rate']:.4f} | {row['schema_pass_rate']:.4f} | "
            f"{row['slot_micro_accuracy']:.4f} | {row['all_fields_exact_rate']:.4f} | "
            f"{row['compiler_outcome_accuracy']:.4f} |"
        )
    (output_dir / "RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def compact_checkpoint(path: Path) -> None:
    """Keep only the latest record per instruction after resumable retries."""

    latest = {row["instruction_id"]: row for row in read_jsonl(path)}
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in latest.values()),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--providers", nargs="+", choices=sorted(PROVIDERS), default=sorted(PROVIDERS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    cases = read_jsonl(args.input)
    if args.limit is not None:
        cases = cases[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_hash = hashlib.sha256((SYSTEM_PROMPT + "\n" + USER_TEMPLATE).encode("utf-8")).hexdigest()
    manifest = {
        "experiment": "SkyRescue-HumanIntent-LLM-v1.0.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "cases": len(cases),
        "providers": {name: {key: value for key, value in PROVIDERS[name].items() if key != "env"} for name in args.providers},
        "parameters": {
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 512,
            "runs_per_case": 1,
            "deepseek_thinking": "disabled",
        },
        "prompt_sha256": prompt_hash,
        "system_prompt": SYSTEM_PROMPT,
        "user_template": USER_TEMPLATE,
        "evaluation_stages": ["direct_json", "schema_validation", "full_skyrescue_compile"],
        "secrets_persisted": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    write_lock = threading.Lock()
    for provider in args.providers:
        key = load_secret(PROVIDERS[provider]["env"], args.key_file)
        raw_path = args.output_dir / f"raw_{provider}.jsonl"
        existing = {
            row["instruction_id"]
            for row in read_jsonl(raw_path)
            if row.get("api_success")
        } if raw_path.exists() else set()
        pending = [case for case in cases if case["instruction_id"] not in existing]
        print(f"{provider}: {len(existing)} cached, {len(pending)} pending", flush=True)
        completed = 0
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {executor.submit(call_one, provider, case, key): case for case in pending}
            for future in as_completed(futures):
                result = future.result()
                with write_lock:
                    with raw_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                completed += 1
                if completed % 10 == 0 or completed == len(pending):
                    print(f"{provider}: completed {completed}/{len(pending)}", flush=True)
        if raw_path.exists():
            compact_checkpoint(raw_path)

    write_outputs(args.output_dir, cases, args.providers)
    print(f"Results written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
