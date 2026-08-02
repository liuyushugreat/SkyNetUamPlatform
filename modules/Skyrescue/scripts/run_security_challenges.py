#!/usr/bin/env python3
"""Evaluate the SkyRescue authorization boundary on frozen challenges."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from skyrescue.security import evaluate_action


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seen, decisions = set(), []
    for request in rows(args.dataset / "requests.jsonl"):
        allowed, reason = evaluate_action(request, seen)
        decisions.append({"request_id": request["request_id"], "allowed": allowed, "reason": reason})
    truth = {row["request_id"]: row for row in rows(args.dataset / "labels.jsonl")}
    correct = sum(decision["allowed"] == truth[decision["request_id"]]["allowed"] for decision in decisions)
    blocked = [d for d in decisions if not truth[d["request_id"]]["allowed"]]
    interception = sum(not d["allowed"] for d in blocked) / len(blocked)
    result = {"count": len(decisions), "accuracy": round(correct / len(decisions), 4), "unauthorized_interception_rate": round(interception, 4), "synthetic_data": True, "notes": ["labels were loaded only after decisions", "this evaluates deterministic policy coverage, not operational flight safety"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
