#!/usr/bin/env python3
"""Generate, score, and summarize a multi-seed fault challenge."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skyrescue.fault_detection import DETECTORS, FAULT_TYPES


METRICS = ("precision", "recall", "f1")
T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def run_checked(command: list[str]) -> None:
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        details = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}\n{details}")


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Cannot summarize an empty sample")
    n = len(values)
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    critical = T_CRITICAL_95.get(n - 1, 1.96)
    margin = critical * std / math.sqrt(n) if n > 1 else 0.0
    return {
        "n": n,
        "mean": round(mean, 6),
        "std": round(std, 6),
        "ci95_low": round(mean - margin, 6),
        "ci95_high": round(mean + margin, 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def paired_sign_flip(differences: list[float]) -> dict[str, float | int | str | None]:
    """Return a two-sided paired sign-flip test on the mean difference."""

    observed = abs(statistics.fmean(differences))
    n = len(differences)
    tolerance = 1e-12
    if n <= 20:
        total = 2**n
        extreme = 0
        for signs in itertools.product((-1, 1), repeat=n):
            statistic = abs(statistics.fmean(sign * value for sign, value in zip(signs, differences)))
            extreme += statistic + tolerance >= observed
        p_value = extreme / total
        test = "exact paired sign-flip"
    else:
        rng = random.Random(20260802)
        total = 200_000
        extreme = 0
        for _ in range(total):
            statistic = abs(
                statistics.fmean(value if rng.random() < 0.5 else -value for value in differences)
            )
            extreme += statistic + tolerance >= observed
        p_value = (extreme + 1) / (total + 1)
        test = "Monte Carlo paired sign-flip"

    diff_summary = summarize(differences)
    diff_std = statistics.stdev(differences) if n > 1 else 0.0
    effect_size = statistics.fmean(differences) / diff_std if diff_std else None
    return {
        "test": test,
        "permutations": total,
        "mean_difference": diff_summary["mean"],
        "ci95_low": diff_summary["ci95_low"],
        "ci95_high": diff_summary["ci95_high"],
        "cohen_dz": round(effect_size, 6) if effect_size is not None else None,
        "p_value": round(p_value, 8),
    }


def holm_adjust(comparisons: dict[str, dict]) -> None:
    ordered = sorted(comparisons.items(), key=lambda item: item[1]["p_value"])
    running_max = 0.0
    count = len(ordered)
    for rank, (_, result) in enumerate(ordered):
        adjusted = min(1.0, result["p_value"] * (count - rank))
        running_max = max(running_max, adjusted)
        result["p_value_holm"] = round(running_max, 8)


def aggregate(payloads: list[dict], methods: list[str], reference: str) -> dict:
    overall = {}
    by_fault_type = {}
    for method in methods:
        overall[method] = {
            metric: summarize([payload["methods"][method]["overall"][metric] for payload in payloads])
            for metric in METRICS
        }
        by_fault_type[method] = {
            fault_type: {
                metric: summarize(
                    [payload["methods"][method]["by_fault_type"][fault_type][metric] for payload in payloads]
                )
                for metric in METRICS
            }
            for fault_type in FAULT_TYPES
        }

    significance = {}
    for metric in METRICS:
        significance[metric] = {}
        reference_values = [payload["methods"][reference]["overall"][metric] for payload in payloads]
        for method in methods:
            if method == reference:
                continue
            baseline_values = [payload["methods"][method]["overall"][metric] for payload in payloads]
            differences = [ref - baseline for ref, baseline in zip(reference_values, baseline_values)]
            significance[metric][method] = paired_sign_flip(differences)
        holm_adjust(significance[metric])

    return {
        "overall": overall,
        "by_fault_type": by_fault_type,
        "paired_significance_vs_reference": significance,
    }


def write_csv_files(output_dir: Path, summary: dict, methods: list[str], reference: str) -> None:
    fields = ["method", "metric", "n", "mean", "std", "ci95_low", "ci95_high", "min", "max"]
    with (output_dir / "overall_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in methods:
            for metric in METRICS:
                writer.writerow({"method": method, "metric": metric, **summary["overall"][method][metric]})

    type_fields = ["method", "fault_type", *fields[1:]]
    with (output_dir / "by_fault_type.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=type_fields)
        writer.writeheader()
        for method in methods:
            for fault_type in FAULT_TYPES:
                for metric in METRICS:
                    writer.writerow(
                        {
                            "method": method,
                            "fault_type": fault_type,
                            "metric": metric,
                            **summary["by_fault_type"][method][fault_type][metric],
                        }
                    )

    significance_fields = [
        "reference",
        "baseline",
        "metric",
        "test",
        "permutations",
        "mean_difference",
        "ci95_low",
        "ci95_high",
        "cohen_dz",
        "p_value",
        "p_value_holm",
    ]
    with (output_dir / "significance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=significance_fields)
        writer.writeheader()
        for metric in METRICS:
            for baseline, result in summary["paired_significance_vs_reference"][metric].items():
                writer.writerow(
                    {"reference": reference, "baseline": baseline, "metric": metric, **result}
                )


def format_mean_std(item: dict) -> str:
    return f"{item['mean']:.4f} ± {item['std']:.4f}"


def write_markdown(
    output_dir: Path,
    summary: dict,
    methods: list[str],
    reference: str,
    seeds: list[int],
    uavs: int,
    duration: int,
    faults: int,
) -> None:
    labels = {
        "single_signal": "Single signal",
        "structural_only": "Structural only",
        "persistent_fusion": "Persistent fusion",
        "skyrescue_fusion": "SkyRescue fusion",
    }
    lines = [
        "# FaultChallenge v1.1.0 十种子实验汇总",
        "",
        f"- 随机种子：{', '.join(str(seed) for seed in seeds)}",
        f"- 每个种子：{uavs} 架无人机，{duration} 秒，{faults} 个注入故障",
        "- 数据性质：完全合成；故障标签仅在检测器推理结束后用于离线评分",
        "- 指标：同一无人机、时间区间重叠且故障类型一致的事件级 Precision/Recall/F1",
        "- 区间：种子间均值的双侧 95% t 置信区间；表中 ± 为样本标准差",
        "",
        "## 总体结果",
        "",
        "| 检测器 | Precision | Recall | F1 | F1 95% CI |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in methods:
        metrics = summary["overall"][method]
        lines.append(
            f"| {labels.get(method, method)} | {format_mean_std(metrics['precision'])} | "
            f"{format_mean_std(metrics['recall'])} | {format_mean_std(metrics['f1'])} | "
            f"[{metrics['f1']['ci95_low']:.4f}, {metrics['f1']['ci95_high']:.4f}] |"
        )

    lines.extend(
        [
            "",
            "## 分故障类型 F1",
            "",
            "| 故障类型 | " + " | ".join(labels.get(method, method) for method in methods) + " |",
            "| --- | " + " | ".join("---:" for _ in methods) + " |",
        ]
    )
    for fault_type in FAULT_TYPES:
        values = [
            format_mean_std(summary["by_fault_type"][method][fault_type]["f1"])
            for method in methods
        ]
        lines.append(f"| {fault_type} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            f"## 配对显著性检验（{labels.get(reference, reference)} 相对基线）",
            "",
            "| 基线 | 平均 ΔF1 | 95% CI | Cohen's dz | 原始 p | Holm 校正 p |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    f1_tests = summary["paired_significance_vs_reference"]["f1"]
    for baseline, result in f1_tests.items():
        effect = "n/a" if result["cohen_dz"] is None else f"{result['cohen_dz']:.3f}"
        lines.append(
            f"| {labels.get(baseline, baseline)} | {result['mean_difference']:.4f} | "
            f"[{result['ci95_low']:.4f}, {result['ci95_high']:.4f}] | {effect} | "
            f"{result['p_value']:.6f} | {result['p_value_holm']:.6f} |"
        )

    lines.extend(
        [
            "",
            "注：显著性检验为双侧精确配对符号翻转检验，Holm 校正在每个指标的三组基线比较内进行。",
            "该结果只能支持合成挑战集上的内部效度，不能替代真实飞行、硬件在环或跨场景外部验证。",
            "",
        ]
    )
    (output_dir / "PAPER_FAULT_RESULTS_zh.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a multi-seed SkyRescue fault challenge")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20261101, 20261111)))
    parser.add_argument("--uavs", type=int, default=20)
    parser.add_argument("--duration", type=int, default=3600)
    parser.add_argument("--faults", type=int, default=120)
    parser.add_argument("--methods", nargs="+", choices=DETECTORS, default=list(DETECTORS))
    parser.add_argument("--reference", choices=DETECTORS, default="skyrescue_fusion")
    args = parser.parse_args()
    if args.reference not in args.methods:
        parser.error("--reference must also be included in --methods")

    script_dir = Path(__file__).resolve().parent
    args.data_dir.mkdir(parents=True, exist_ok=True)
    per_seed_dir = args.output_dir / "per_seed"
    per_seed_dir.mkdir(parents=True, exist_ok=True)

    payloads = []
    for index, seed in enumerate(args.seeds, start=1):
        dataset = args.data_dir / f"fault_challenge_seed_{seed}"
        result_path = per_seed_dir / f"fault_challenge_seed_{seed}.json"
        print(f"[{index}/{len(args.seeds)}] Generating seed {seed}", flush=True)
        run_checked(
            [
                sys.executable,
                str(script_dir / "generate_fault_challenge.py"),
                "--output",
                str(dataset),
                "--seed",
                str(seed),
                "--uavs",
                str(args.uavs),
                "--duration",
                str(args.duration),
                "--faults",
                str(args.faults),
            ]
        )
        print(f"[{index}/{len(args.seeds)}] Scoring seed {seed}", flush=True)
        run_checked(
            [
                sys.executable,
                str(script_dir / "run_fault_challenge.py"),
                "--dataset",
                str(dataset),
                "--output",
                str(result_path),
                "--methods",
                *args.methods,
            ]
        )
        payloads.append(json.loads(result_path.read_text(encoding="utf-8")))

    aggregates = aggregate(payloads, args.methods, args.reference)
    summary = {
        "benchmark": "SkyRescue-FaultChallenge",
        "version": "1.1.0",
        "synthetic_data": True,
        "seeds": args.seeds,
        "configuration": {
            "uavs_per_seed": args.uavs,
            "duration_s_per_seed": args.duration,
            "faults_per_seed": args.faults,
            "total_faults": args.faults * len(args.seeds),
        },
        "methods": args.methods,
        "reference_method": args.reference,
        "metric_definition": (
            "typed event-level overlap: same UAV, overlapping time interval, and same fault_type"
        ),
        "confidence_interval": "two-sided 95% t interval over random seeds",
        "significance_test": "two-sided paired sign-flip test with Holm correction per metric",
        **aggregates,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv_files(args.output_dir, summary, args.methods, args.reference)
    write_markdown(
        args.output_dir,
        summary,
        args.methods,
        args.reference,
        args.seeds,
        args.uavs,
        args.duration,
        args.faults,
    )
    print(f"Wrote multi-seed summary to {args.output_dir}")


if __name__ == "__main__":
    main()
