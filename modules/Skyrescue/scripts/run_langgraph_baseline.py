#!/usr/bin/env python3
"""Run the fair LangGraph runtime baseline on the frozen intent benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skyrescue.langgraph_baseline import evaluate_langgraph_runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = [json.loads(line) for line in (args.dataset / "intent_cases.jsonl").read_text(encoding="utf-8").splitlines() if line]
    metrics = evaluate_langgraph_runtime(cases, args.output_dir / "langgraph_checkpoints.sqlite")
    payload = {"dataset": str(args.dataset), "metrics": metrics}
    (args.output_dir / "langgraph_workflow_baseline.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
