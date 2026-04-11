"""ExplanationAgent: generates rule-grounded, traceable natural language explanations.

Supports four prompt modes for ablation study:
  - full:           rule list injection + format constraint (default)
  - rule_list_only: rule list injection without format constraint
  - format_only:    format constraint without rule list injection
  - generic:        vanilla prompt without any constraint
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .base_agent import BaseAgent, AgentResult, AgentVerdict, TraceEntry

PROMPT_MODES = ("full", "rule_list_only", "format_only", "generic")


class ExplanationAgent(BaseAgent):
    """Generates human-readable explanations with mandatory regulation citations.

    Takes the structured outputs from ComplianceAgent and RiskAssessmentAgent,
    then produces a unified explanation text where every claim is linked to a
    specific regulation clause ID.
    """

    name = "explanation"

    def __init__(self, config=None, llm_client=None, prompt_mode: str = "full"):
        super().__init__(config)
        self.llm_client = llm_client
        if prompt_mode not in PROMPT_MODES:
            raise ValueError(f"prompt_mode must be one of {PROMPT_MODES}, got '{prompt_mode}'")
        self.prompt_mode = prompt_mode

    KNOWN_RULES = (
        "REG-WIND-001（风速超限）、REG-BAT-001（电量不足）、"
        "REG-ALT-001（超高度限制）、REG-VIS-001（能见度不足）、"
        "REG-LOAD-001（载荷超重）、REG-SPEED-001（速度超限）、"
        "REG-TEMP-001（温度超限）、REG-ZONE-001（限制空域）、"
        "REG-SAFETY-012（综合安全评估）、REG-SAFETY-013（应急措施建议）"
    )

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        compliance_result = context.get("compliance_result")
        risk_result = context.get("risk_result")
        scenario = context.get("scenario", {})
        is_retry = context.get("is_retry", False)
        audit_feedback = context.get("audit_feedback")

        prompt = self._build_prompt(
            scenario, compliance_result, risk_result,
            is_retry=is_retry, audit_feedback=audit_feedback,
        )
        raw_explanation = self._call_llm(prompt)
        cited_rules = self._extract_rule_ids(raw_explanation)

        traces = [
            TraceEntry(
                step="explanation_generation",
                source=self.name,
                rule_ids=cited_rules,
                detail=f"Generated explanation citing {len(cited_rules)} rule(s), prompt_mode={self.prompt_mode}",
            )
        ]

        overall_verdict = AgentVerdict.SAFE
        if compliance_result and compliance_result.get("verdict") == "violation":
            overall_verdict = AgentVerdict.VIOLATION
        elif risk_result and risk_result.get("risk_level") in ("high", "medium"):
            overall_verdict = AgentVerdict.RISK

        return AgentResult(
            agent_name=self.name,
            verdict=overall_verdict,
            confidence=risk_result.get("confidence", 0.5) if risk_result else 0.5,
            payload={
                "explanation": raw_explanation,
                "cited_rules": cited_rules,
                "prompt_mode": self.prompt_mode,
            },
            traces=traces,
        )

    def _build_prompt(
        self,
        scenario: Dict[str, Any],
        compliance: Dict[str, Any] | None,
        risk: Dict[str, Any] | None,
        *,
        is_retry: bool = False,
        audit_feedback: Dict[str, Any] | None = None,
    ) -> str:
        parts = ["请根据以下合规检查结果和风险评估结果，生成一段治理决策解释。", ""]

        if self.prompt_mode in ("full", "format_only"):
            parts.extend([
                "【格式规范——必须严格遵守】",
                "1. 每一句话（以'。'结尾）必须包含至少一个法规编号（如 REG-WIND-001）。",
                "2. 每句话以'根据REG-XXX-NNN，'或'依据REG-XXX-NNN，'开头。",
                "3. 禁止出现没有法规编号的判断性语句。",
                "4. 末尾必须附上'【引用条款】'汇总所有引用的法规编号。",
                "",
            ])

        if self.prompt_mode in ("full", "rule_list_only"):
            parts.extend([
                f"【可引用法规编号】\n{self.KNOWN_RULES}",
                "",
            ])

        if self.prompt_mode in ("full", "format_only"):
            parts.extend([
                "【输出示例】",
                "根据REG-WIND-001，当前环境风速超过该型号UAV最大抗风等级，构成风速超限违规。",
                "依据REG-SAFETY-012，综合评估后建议提高监控等级。",
                "【引用条款】REG-WIND-001, REG-SAFETY-012",
                "",
            ])

        parts.append(f"【飞行场景】\n{json.dumps(scenario, ensure_ascii=False, default=str)}")
        if compliance:
            parts.append(
                f"\n【合规检查结果】\n{json.dumps(compliance, ensure_ascii=False, default=str)}"
            )
        if risk:
            parts.append(
                f"\n【风险评估结果】\n{json.dumps(risk, ensure_ascii=False, default=str)}"
            )
        if is_retry and audit_feedback:
            parts.append(
                f"\n【审计未通过——请改进引用密度】\n"
                f"上次解释的 RAR 评分仅为 {audit_feedback.get('rar', 'N/A')}（需≥0.8），"
                f"UCR 评分为 {audit_feedback.get('ucr', 'N/A')}（需≤0.1）。\n"
                f"请严格按照上述格式，让每一句都以「根据 REG-XXX-NNN，」开头。"
            )
        return "\n".join(parts)

    def _build_system_prompt(self) -> str:
        """Build the system prompt based on current prompt_mode."""
        if self.prompt_mode == "generic":
            return (
                "你是SkyGov低空交通合规治理系统的解释生成模块，"
                "请根据飞行场景的合规检查和风险评估结果生成治理决策解释。"
            )

        base = "你是SkyGov低空交通合规治理系统的解释生成模块，输出直接呈现给民航监管人员审阅。\n"

        if self.prompt_mode in ("full", "format_only"):
            base += (
                "你必须严格遵守以下输出规范：\n"
                "1. 每一个判断句（以句号结尾）必须包含至少一个法规编号。\n"
                "2. 法规编号格式：REG-大写字母-三位数字（如REG-WIND-001）。\n"
                "3. 每句话以'依据REG-XXX-NNN，'或'根据REG-XXX-NNN，'开头。\n"
                "4. 绝对禁止出现没有法规编号的判断性语句。\n"
                "5. 末尾用【引用条款】汇总所有引用的法规编号。\n"
            )

        if self.prompt_mode in ("full", "rule_list_only"):
            base += f"6. 可引用法规：{self.KNOWN_RULES}。"

        return base

    def _call_llm(self, prompt: str) -> str:
        if self.llm_client is None:
            return self._mock_explanation()
        try:
            response = self.llm_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                stream=False,
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error("Explanation LLM call failed: %s", e)
            return f"[系统错误] 解释生成失败：{e}"

    def _extract_rule_ids(self, text: str) -> List[str]:
        import re
        return list(set(re.findall(r"REG-[A-Z]+-\d+", text)))

    def _mock_explanation(self) -> str:
        return (
            "根据REG-SAFETY-013，综合SkyGov治理系统评估，本次飞行任务决策如下。\n"
            "依据REG-WIND-001，当前环境风速超过该型号UAV最大抗风等级，构成风速超限违规。\n"
            "依据REG-BAT-001，当前电池电量需确认是否高于15%安全阈值。\n"
            "根据REG-SAFETY-012，风速超限条件下飞行姿态稳定性显著下降，坠机风险评级为「高」。\n"
            "根据REG-ALT-001，飞行高度须控制在空域限高范围内。\n"
            "根据REG-LOAD-001，当前载荷应在UAV最大载荷能力范围内运行。\n"
            "依据REG-SAFETY-013，综合以上条款，建议立即执行返航或迫降程序。\n"
            "【引用条款】REG-WIND-001, REG-BAT-001, REG-SAFETY-012, REG-ALT-001, REG-LOAD-001, REG-SAFETY-013"
        )
