"""RiskAssessmentAgent: KG-retrieval + LLM semantic risk reasoning (extends SkyKG RAG)."""

from __future__ import annotations

import json
from typing import Any, Dict

from .base_agent import BaseAgent, AgentResult, AgentVerdict, TraceEntry


class RiskAssessmentAgent(BaseAgent):
    """Performs semantic risk reasoning via RAG over the UAM knowledge graph.

    Workflow:
        1. Receive flight context (UAV profile, airspace, weather, mission).
        2. Retrieve relevant KG facts and regulation chunks via RAG pipeline.
        3. Construct a grounded prompt with retrieved evidence.
        4. Call LLM to assess risk level and identify potential semantic risks.
        5. Return structured risk assessment with evidence citations.
    """

    name = "risk_assessment"

    def __init__(self, config=None, retriever=None, llm_client=None):
        super().__init__(config)
        self.retriever = retriever
        self.llm_client = llm_client

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        uav_id = context.get("uav_id", "UNKNOWN")
        telemetry = context.get("telemetry", {})
        mission = context.get("mission", {})

        retrieved_chunks = self._retrieve(context)
        prompt = self._build_prompt(context, retrieved_chunks)
        llm_output = self._call_llm(prompt)
        risk_level, parsed = self._parse_response(llm_output)

        verdict_map = {
            "high": AgentVerdict.RISK,
            "medium": AgentVerdict.UNCERTAIN,
            "low": AgentVerdict.SAFE,
            "none": AgentVerdict.SAFE,
        }
        verdict = verdict_map.get(risk_level, AgentVerdict.UNCERTAIN)
        confidence = parsed.get("confidence", 0.5)

        traces = [
            TraceEntry(
                step="rag_retrieval",
                source=self.name,
                detail=f"Retrieved {len(retrieved_chunks)} chunks",
            ),
            TraceEntry(
                step="llm_risk_inference",
                source=self.name,
                rule_ids=parsed.get("cited_rules", []),
                detail=f"Risk level: {risk_level}",
            ),
        ]

        return AgentResult(
            agent_name=self.name,
            verdict=verdict,
            confidence=confidence,
            payload={
                "risk_level": risk_level,
                "risk_factors": parsed.get("risk_factors", []),
                "cited_rules": parsed.get("cited_rules", []),
                "raw_explanation": parsed.get("explanation", ""),
                "retrieved_chunk_count": len(retrieved_chunks),
            },
            traces=traces,
        )

    def _retrieve(self, context: Dict[str, Any]):
        """Retrieve relevant KG facts and regulation chunks."""
        if self.retriever is None:
            return self._mock_retrieval(context)
        return self.retriever.retrieve(context)

    KNOWN_RULES = (
        "REG-WIND-001, REG-BAT-001, REG-ALT-001, REG-VIS-001, "
        "REG-LOAD-001, REG-SPEED-001, REG-TEMP-001, REG-ZONE-001, "
        "REG-SAFETY-012, REG-SAFETY-013"
    )

    def _build_prompt(self, context: Dict[str, Any], chunks: list) -> str:
        evidence = "\n".join(
            f"[{i + 1}] {c.get('text', str(c))}" for i, c in enumerate(chunks)
        )
        return (
            "你是一个低空交通安全评估专家AI。根据以下检索到的知识图谱事实和法规条款，"
            "评估该飞行场景的风险等级。\n\n"
            "【飞行场景】\n"
            f"{json.dumps(context, ensure_ascii=False, default=str)}\n\n"
            "【检索证据】\n"
            f"{evidence}\n\n"
            f"【可引用法规编号】\n{self.KNOWN_RULES}\n\n"
            "【输出要求】\n"
            "请以JSON格式输出。explanation 字段中每句话必须引用至少一个 REG-XXX-NNN 法规编号：\n"
            '{"risk_level": "high/medium/low/none", '
            '"confidence": 0.0-1.0, '
            '"risk_factors": ["..."], '
            '"cited_rules": ["REG-WIND-001", "REG-BAT-001"], '
            '"explanation": "根据REG-WIND-001，...。根据REG-BAT-001，...。"}'
        )

    def _call_llm(self, prompt: str) -> str:
        if self.llm_client is None:
            return self._mock_llm(prompt)
        try:
            response = self.llm_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是SkyGov低空交通治理系统的风险评估Agent。\n"
                            "你的所有判断必须严格基于检索到的知识图谱事实和法规条款，"
                            "禁止使用未在证据中出现的信息。\n"
                            "在 explanation 字段中，每句话必须引用至少一个法规编号"
                            "（格式：REG-XXX-NNN），每句以'根据REG-XXX-NNN，'开头。\n"
                            f"可引用法规：{self.KNOWN_RULES}。"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                stream=False,
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error("LLM call failed: %s", e)
            return json.dumps(
                {"risk_level": "uncertain", "confidence": 0.0, "explanation": str(e)}
            )

    def _parse_response(self, raw: str):
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            parsed = json.loads(raw[start:end])
            return parsed.get("risk_level", "uncertain"), parsed
        except (ValueError, json.JSONDecodeError):
            return "uncertain", {"explanation": raw, "confidence": 0.0}

    def _mock_retrieval(self, context: Dict[str, Any]):
        """Fallback retrieval returning synthetic evidence chunks."""
        uav_id = context.get("uav_id", "UAV-UNKNOWN")
        telemetry = context.get("telemetry", {})
        return [
            {"text": f"根据REG-WIND-001：UAV {uav_id} 最大抗风等级为{telemetry.get('wind_resistance', 5)}级，超限应返航", "rule_id": "REG-WIND-001"},
            {"text": "根据REG-BAT-001：电池安全阈值为15%，低于此阈值应强制降落或返航", "rule_id": "REG-BAT-001"},
            {"text": f"根据REG-ALT-001：空域最大允许高度为{telemetry.get('max_altitude', 300)}米，超高触发违规", "rule_id": "REG-ALT-001"},
            {"text": "根据REG-VIS-001：最低能见度要求1.5公里，低于此标准禁止目视飞行", "rule_id": "REG-VIS-001"},
            {"text": f"根据REG-LOAD-001：最大载荷{telemetry.get('max_payload_kg', 10)}kg，超载存在结构风险", "rule_id": "REG-LOAD-001"},
            {"text": f"根据REG-SPEED-001：最大允许速度{telemetry.get('max_speed_ms', 20)}m/s，超速增加碰撞风险", "rule_id": "REG-SPEED-001"},
            {"text": "根据REG-TEMP-001：安全工作温度-20°C至45°C，超出范围电池性能下降", "rule_id": "REG-TEMP-001"},
            {"text": "根据REG-SAFETY-012：多风险因素叠加时应提升监控等级", "rule_id": "REG-SAFETY-012"},
            {"text": "根据REG-SAFETY-013：存在违规或高风险时应执行返航或迫降等应急措施", "rule_id": "REG-SAFETY-013"},
        ]

    def _mock_llm(self, prompt: str) -> str:
        return json.dumps(
            {
                "risk_level": "medium",
                "confidence": 0.75,
                "risk_factors": ["环境风速接近机型抗风上限", "任务载荷较重增加能耗", "飞行高度接近限高"],
                "cited_rules": ["REG-WIND-001", "REG-BAT-001", "REG-SAFETY-012", "REG-ALT-001", "REG-LOAD-001"],
                "explanation": (
                    "根据REG-WIND-001，该UAV当前飞行环境风速接近其型号抗风上限。"
                    "根据REG-BAT-001，电池电量需确认高于15%安全阈值。"
                    "根据REG-ALT-001，飞行高度需在限高范围内。"
                    "根据REG-SAFETY-012，综合评估建议提高监控等级。"
                    "根据REG-LOAD-001，载荷应在允许范围内飞行。"
                ),
            },
            ensure_ascii=False,
        )
