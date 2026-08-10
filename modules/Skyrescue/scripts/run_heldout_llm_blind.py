#!/usr/bin/env python3
"""Capture frozen DeepSeek/Qwen responses for an unlabeled held-out set."""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_human_intent_llm_benchmark import (
    PROVIDERS,
    SYSTEM_PROMPT as DEVELOPMENT_SYSTEM_PROMPT,
    compact_checkpoint,
    load_secret,
    post_json,
    read_jsonl,
    schema_errors,
    strict_json,
)
from skyrescue.entity_grounding import compile_grounded_candidate
from skyrescue.workflow import compile_generated_candidate


EXPERIMENT = "SkyRescue-HeldOut100-InstructionOnly-Blind-v1.0.0"
PROMPT_VERSION = "SkyRescue-IntentPrompt-InstructionOnly-v1.0.0"
SYSTEM_PROMPT = DEVELOPMENT_SYSTEM_PROMPT.replace(
    "请依据场景背景和指挥指令抽取一个结构化任务。",
    "请仅依据指挥指令抽取一个结构化任务。",
)
USER_TEMPLATE = "指挥指令：{instruction_text}"
PARAMETERS = {
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 512,
    "runs_per_case": 1,
    "deepseek_thinking": "disabled",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def prompt_sha256() -> str:
    return sha256_bytes((SYSTEM_PROMPT + "\n" + USER_TEMPLATE).encode("utf-8"))


def validate_cases(cases: list[dict[str, Any]]) -> None:
    if not cases:
        raise ValueError("The held-out input is empty.")
    identifiers = []
    for row in cases:
        if set(row) != {"instruction_id", "instruction_text"}:
            raise ValueError("Blind input must contain only instruction_id and instruction_text.")
        instruction_id = str(row["instruction_id"]).strip()
        instruction_text = str(row["instruction_text"]).strip()
        if not instruction_id or len(instruction_text) < 8:
            raise ValueError(f"Malformed blind-input row: {instruction_id or '<missing>'}")
        identifiers.append(instruction_id)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Blind input contains duplicate instruction IDs.")


def build_user_prompt(case: dict[str, Any]) -> str:
    return USER_TEMPLATE.format(instruction_text=str(case["instruction_text"]).strip())


def call_one(provider: str, case: dict[str, Any], key: str) -> dict[str, Any]:
    config = PROVIDERS[provider]
    user_prompt = build_user_prompt(case)
    payload: dict[str, Any] = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": PARAMETERS["temperature"],
        "top_p": PARAMETERS["top_p"],
        "max_tokens": PARAMETERS["max_tokens"],
        "stream": False,
    }
    if provider == "deepseek":
        payload["thinking"] = {"type": "disabled"}

    started = time.perf_counter()
    error = None
    response = None
    attempts = 0
    for attempts in range(1, 5):
        try:
            response = post_json(config["url"], key, payload)
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            error = f"HTTP {exc.code}: {body}"
            if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempts == 4:
                break
        except Exception as exc:  # pragma: no cover - network-dependent
            error = f"{type(exc).__name__}: {exc}"
            if attempts == 4:
                break
        time.sleep(2 ** (attempts - 1))

    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    base = {
        "experiment": EXPERIMENT,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256(),
        "provider": provider,
        "requested_model": config["model"],
        "instruction_id": case["instruction_id"],
        "instruction_sha256": sha256_bytes(str(case["instruction_text"]).encode("utf-8")),
        "input_fields": ["instruction_text"],
        "scenario_card_sent": False,
        "gold_labels_sent": False,
        "attempts": attempts,
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
    compile_result = None
    grounded_compile_result = None
    if parsed is not None and not errors:
        compile_result = compile_generated_candidate(parsed).to_dict()
        grounded_compile_result = compile_grounded_candidate(
            parsed,
            "",
            str(case["instruction_text"]),
        ).to_dict()
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
        "compile_result": compile_result,
        "grounded_compile_result": grounded_compile_result,
    }


def write_capture_summary(output_dir: Path, providers: list[str]) -> None:
    rows = []
    for provider in providers:
        path = output_dir / f"raw_{provider}.jsonl"
        records = read_jsonl(path) if path.exists() else []
        api_success = sum(bool(row.get("api_success")) for row in records)
        parsed = sum(bool(row.get("json_parse_success")) for row in records)
        schema = sum(bool(row.get("schema_valid")) for row in records)
        compiled = sum(bool((row.get("compile_result") or {}).get("executable")) for row in records)
        grounded = sum(
            bool(((row.get("grounded_compile_result") or {}).get("compilation") or {}).get("executable"))
            for row in records
        )
        grounding_gate = sum(
            ((row.get("grounded_compile_result") or {}).get("compilation") or {}).get("failure") == "UngroundedEntity"
            for row in records
        )
        count = len(records)
        rows.append({
            "provider": provider,
            "requested_model": PROVIDERS[provider]["model"],
            "captured": count,
            "api_success": api_success,
            "direct_json": parsed,
            "schema_valid": schema,
            "direct_compile_executable": compiled,
            "grounded_compile_executable": grounded,
            "ungrounded_entity_gate": grounding_gate,
        })
    (output_dir / "capture_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
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
    validate_cases(cases)
    if args.limit is not None:
        cases = cases[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": EXPERIMENT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input.resolve()),
        "input_sha256": sha256_bytes(args.input.read_bytes()),
        "cases": len(cases),
        "input_fields": ["instruction_id", "instruction_text"],
        "scenario_card_sent": False,
        "gold_labels_sent": False,
        "providers": {
            name: {key: value for key, value in PROVIDERS[name].items() if key != "env"}
            for name in args.providers
        },
        "parameters": PARAMETERS,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256(),
        "system_prompt": SYSTEM_PROMPT,
        "user_template": USER_TEMPLATE,
        "evaluation_stages": [
            "direct_json",
            "schema_validation",
            "full_skyrescue_compile",
            "frozen_entity_grounding_gate",
        ],
        "gold_scoring_status": "locked_until_a_b_annotation_and_adjudication",
        "secrets_persisted": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
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

    write_capture_summary(args.output_dir, args.providers)
    print(f"Blind responses written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
