"""Compliance report generator: produces auditable governance decision reports."""

from __future__ import annotations

import json
import time
from typing import Any, Dict

from .decision_tracer import DecisionRecord


class ComplianceReportGenerator:
    """Generates structured compliance reports from decision records.

    Output formats:
        - JSON (machine-readable, for downstream systems)
        - Markdown (human-readable, for regulatory review)
    """

    def to_json(self, record: DecisionRecord) -> str:
        return record.to_json()

    def to_markdown(self, record: DecisionRecord) -> str:
        lines = [
            f"# 低空交通合规治理决策报告",
            "",
            f"**请求ID**: {record.request_id}",
            f"**UAV编号**: {record.uav_id}",
            f"**时间**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(record.timestamp))}",
            "",
            "## 飞行场景",
            f"```json\n{json.dumps(record.scenario, ensure_ascii=False, indent=2, default=str)}\n```",
            "",
            "## Agent推理链路",
        ]

        for i, step in enumerate(record.agent_chain, 1):
            agent = step.get("agent", "unknown")
            verdict = step.get("verdict", "N/A")
            lines.append(f"### {i}. {agent}")
            lines.append(f"- **判定**: {verdict}")
            if "payload" in step:
                lines.append(f"- **详情**: {json.dumps(step['payload'], ensure_ascii=False, default=str)}")
            lines.append("")

        lines.extend([
            "## 最终决策",
            f"- **裁定**: {record.final_verdict}",
            f"- **执行动作**: {record.final_action}",
            f"- **引用条款**: {', '.join(record.cited_rules) if record.cited_rules else '无'}",
            "",
            "## 解释文本",
            record.explanation or "（无）",
            "",
            "## 质量评分",
        ])

        for metric, score in record.quality_scores.items():
            lines.append(f"- **{metric.upper()}**: {score:.4f}")

        lines.append("\n---\n*本报告由SkyGov多智能体治理系统自动生成*")
        return "\n".join(lines)
