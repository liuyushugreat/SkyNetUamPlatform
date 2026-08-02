#!/usr/bin/env python3
"""Score weak-signal fault detectors with labels opened only after inference."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skyrescue.fault_detection import DETECTORS, FAULT_TYPES, detect


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def overlaps(a: dict, b: dict, typed: bool = True) -> bool:
    same_type = (not typed) or a.get("fault_type") == b.get("fault_type")
    return (
        same_type
        and a["uav_id"] == b["uav_id"]
        and a["start_time_s"] < b["end_time_s"]
        and b["start_time_s"] < a["end_time_s"]
    )


def prf(matched_predictions: int, predicted: int, hits: int, truth: int) -> dict[str, float | int]:
    precision = matched_predictions / max(1, predicted)
    recall = hits / max(1, truth)
    f1 = 2 * precision * recall / max(0.0001, precision + recall)
    return {
        "predicted": predicted,
        "truth": truth,
        "matched_predictions": matched_predictions,
        "hits": hits,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def score_predictions(predictions: list[dict], truth: list[dict]) -> dict:
    matched = sum(any(overlaps(prediction, fault) for fault in truth) for prediction in predictions)
    hits = sum(any(overlaps(fault, prediction) for prediction in predictions) for fault in truth)
    by_type = {}
    for fault_type in FAULT_TYPES:
        type_predictions = [item for item in predictions if item.get("fault_type") == fault_type]
        type_truth = [item for item in truth if item.get("fault_type") == fault_type]
        type_matched = sum(any(overlaps(prediction, fault) for fault in type_truth) for prediction in type_predictions)
        type_hits = sum(any(overlaps(fault, prediction) for prediction in type_predictions) for fault in type_truth)
        by_type[fault_type] = prf(type_matched, len(type_predictions), type_hits, len(type_truth))

    untyped_matched = sum(any(overlaps(prediction, fault, typed=False) for fault in truth) for prediction in predictions)
    untyped_hits = sum(any(overlaps(fault, prediction, typed=False) for prediction in predictions) for fault in truth)
    return {
        "overall": prf(matched, len(predictions), hits, len(truth)),
        "overall_untyped_overlap": prf(untyped_matched, len(predictions), untyped_hits, len(truth)),
        "by_fault_type": by_type,
    }


def merge_predictions(predictions: list[dict], max_gap_s: int = 5) -> list[dict]:
    """Merge fragmented online detections into event-level predictions."""

    merged: list[dict] = []
    ordered = sorted(
        predictions,
        key=lambda item: (item["uav_id"], item.get("fault_type", "unknown"), item["start_time_s"], item["end_time_s"]),
    )
    for prediction in ordered:
        if not merged:
            merged.append(dict(prediction))
            continue
        previous = merged[-1]
        same_event_stream = (
            previous["uav_id"] == prediction["uav_id"]
            and previous.get("fault_type") == prediction.get("fault_type")
            and prediction["start_time_s"] <= previous["end_time_s"] + max_gap_s
        )
        if same_event_stream:
            previous["end_time_s"] = max(previous["end_time_s"], prediction["end_time_s"])
            previous["signals"] = sorted(set(previous.get("signals", [])) | set(prediction.get("signals", [])))
        else:
            merged.append(dict(prediction))
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="SkyRescue fault-challenge scorer")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=DETECTORS, default=list(DETECTORS))
    args = parser.parse_args()

    telemetry = list(rows(args.dataset / "telemetry.jsonl"))
    truth = list(rows(args.dataset / "faults.jsonl"))
    scores = {}
    predictions_by_method = {}
    for method in args.methods:
        raw_predictions = list(detect(iter(telemetry), method=method))
        predictions = merge_predictions(raw_predictions)
        predictions_by_method[method] = predictions
        scores[method] = score_predictions(predictions, truth)
        scores[method]["raw_predicted_intervals"] = len(raw_predictions)

    truth_by_type = defaultdict(int)
    for fault in truth:
        truth_by_type[fault["fault_type"]] += 1

    result = {
        "dataset": str(args.dataset),
        "faults": len(truth),
        "truth_by_fault_type": dict(sorted(truth_by_type.items())),
        "methods": scores,
        "prediction_examples": {
            method: predictions[:5] for method, predictions in predictions_by_method.items()
        },
        "synthetic_data": True,
        "truth_separation": "fault labels are opened only after detector inference",
        "metric_definition": "typed event-level overlap: same UAV, overlapping time interval, and same fault_type",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
