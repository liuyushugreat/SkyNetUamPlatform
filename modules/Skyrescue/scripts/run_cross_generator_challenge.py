#!/usr/bin/env python3
"""Run frozen SkyRescue detectors on an independent generator family."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_fault_challenge_multiseed import (  # noqa: E402
    DETECTORS,
    FAULT_TYPES,
    aggregate,
    format_mean_std,
    write_csv_files,
)


def run_checked(command: list[str]) -> None:
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        details = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}\n{details}")


def write_markdown(output_dir: Path, summary: dict) -> None:
    labels = {
        "single_signal": "Single signal",
        "structural_only": "Structural only",
        "persistent_fusion": "Persistent fusion",
        "skyrescue_fusion": "SkyRescue fusion",
    }
    methods = summary["methods"]
    lines = [
        "# SkyRescue-CrossGenerator v1.0.0 十种子实验汇总",
        "",
        f"- 随机种子：{', '.join(str(seed) for seed in summary['seeds'])}",
        f"- 总故障数：{summary['configuration']['total_faults']}",
        "- 分布迁移：异质基线、AR(1) 相关噪声、渐变故障包络、间歇可观测性和群组相关良性扰动",
        "- 冻结策略：检测器阈值与 FaultChallenge v1.1.0 完全相同，未在本挑战上重新调参",
        "- 指标：同一 UAV、时间区间重叠且故障类型一致的事件级 Precision/Recall/F1",
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
            "## 配对显著性检验（SkyRescue fusion 相对基线）",
            "",
            "| 基线 | 平均 ΔF1 | 95% CI | Cohen's dz | 原始 p | Holm 校正 p |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for baseline, result in summary["paired_significance_vs_reference"]["f1"].items():
        effect = "n/a" if result["cohen_dz"] is None else f"{result['cohen_dz']:.3f}"
        lines.append(
            f"| {labels.get(baseline, baseline)} | {result['mean_difference']:.4f} | "
            f"[{result['ci95_low']:.4f}, {result['ci95_high']:.4f}] | {effect} | "
            f"{result['p_value']:.6f} | {result['p_value_holm']:.6f} |"
        )
    lines.extend(
        [
            "",
            "该挑战仍是合成实验，但生成机制与内部挑战不同，用于检验冻结检测器在分布迁移下的可迁移性。",
            "它不能替代真实飞行或硬件在环验证。",
            "",
        ]
    )
    (output_dir / "PAPER_CROSS_GENERATOR_RESULTS_zh.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SkyRescue cross-generator challenge")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20261201, 20261211)))
    parser.add_argument("--uavs", type=int, default=20)
    parser.add_argument("--duration", type=int, default=3600)
    parser.add_argument("--faults", type=int, default=120)
    parser.add_argument("--methods", nargs="+", choices=DETECTORS, default=list(DETECTORS))
    parser.add_argument("--reference", choices=DETECTORS, default="skyrescue_fusion")
    args = parser.parse_args()
    if args.reference not in args.methods:
        parser.error("--reference must also be included in --methods")

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_dir = args.output_dir / "per_seed"
    per_seed_dir.mkdir(parents=True, exist_ok=True)
    payloads = []
    for index, seed in enumerate(args.seeds, start=1):
        dataset = args.data_dir / f"cross_generator_seed_{seed}"
        result_path = per_seed_dir / f"cross_generator_seed_{seed}.json"
        print(f"[{index}/{len(args.seeds)}] Generating cross-generator seed {seed}", flush=True)
        run_checked(
            [
                sys.executable,
                str(SCRIPT_DIR / "generate_cross_generator_challenge.py"),
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
                str(SCRIPT_DIR / "run_fault_challenge.py"),
                "--dataset",
                str(dataset),
                "--output",
                str(result_path),
                "--methods",
                *args.methods,
            ]
        )
        payloads.append(json.loads(result_path.read_text(encoding="utf-8")))

    summary = {
        "benchmark": "SkyRescue-CrossGenerator",
        "version": "1.0.0",
        "synthetic_data": True,
        "generator_family": "heterogeneous autoregressive state-space",
        "frozen_detector_thresholds": True,
        "source_detector_configuration": "SkyRescue-FaultChallenge v1.1.0",
        "seeds": args.seeds,
        "configuration": {
            "uavs_per_seed": args.uavs,
            "duration_s_per_seed": args.duration,
            "faults_per_seed": args.faults,
            "total_faults": args.faults * len(args.seeds),
        },
        "methods": args.methods,
        "reference_method": args.reference,
        "metric_definition": "typed event-level overlap: same UAV, overlapping interval, same fault_type",
        "confidence_interval": "two-sided 95% t interval over random seeds",
        "significance_test": "two-sided paired sign-flip test with Holm correction per metric",
        **aggregate(payloads, args.methods, args.reference),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv_files(args.output_dir, summary, args.methods, args.reference)
    write_markdown(args.output_dir, summary)
    print(f"Wrote cross-generator summary to {args.output_dir}")


if __name__ == "__main__":
    main()
