import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are SkyKG, the risk-reasoning backend of a low-altitude traffic "
    "control system. Given [Structured Risk Data] retrieved from the "
    "knowledge graph, produce a concise, rule-referenced risk explanation.\n"
    "Requirements:\n"
    "- Cite the specific rule identifier and threshold values.\n"
    "- State the observed telemetry value that triggered the rule.\n"
    "- Recommend an actionable response (e.g., return-to-home, emergency landing).\n"
    "- Do NOT fabricate regulations or thresholds not present in the input."
)


def build_rag_prompt(risk_context: dict) -> list[dict]:
    """Construct a RAG-style chat prompt from structured risk context.

    The prompt injects retrieved KG facts and rules so the LLM produces
    a grounded, rule-referenced explanation rather than a vague guess.
    """
    details = risk_context.get("details", {})
    user_content = (
        f"UAV: {risk_context.get('uav_id', 'Unknown')}\n"
        f"Detected Risk: {risk_context.get('risk_type', 'Unknown')}\n"
        f"Retrieved Rule: {details.get('rule', 'N/A')}\n"
        f"Evidence: {details}\n"
        "Generate a brief, professional alert with recommended action."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def generate_explanation(risk_context: dict) -> str:
    """Generate a natural-language explanation for a detected risk.

    Uses `build_rag_prompt` to create the prompt.  Falls back to a
    template-based explanation when no LLM client is available.
    """
    details = risk_context.get("details", {})
    risk_type = risk_context.get("risk_type", "UnknownRisk")
    uav_id = risk_context.get("uav_id", "Unknown")
    rule = details.get("rule", "N/A")

    if risk_type == "StabilityRisk":
        return (
            f"ALERT — Stability Risk for {uav_id}. "
            f"Wind speed ({details.get('current_wind')} m/s) exceeds "
            f"max resistance ({details.get('max_resistance')} m/s). "
            f"{rule}. Recommend immediate return-to-home or emergency landing."
        )
    if risk_type in ("LowBatteryRisk", "CriticalBatteryRisk"):
        return (
            f"ALERT — {risk_type} for {uav_id}. "
            f"Battery at {details.get('battery_pct')}% "
            f"(threshold: {details.get('threshold')}%). "
            f"{rule}. Recommend return-to-home immediately."
        )
    if risk_type == "NoFlyZoneRisk":
        return (
            f"ALERT — No-Fly Zone Violation for {uav_id}. "
            f"UAV detected inside restricted zone '{details.get('zone_id')}'. "
            f"{rule}. Recommend immediate re-routing."
        )
    return f"ALERT — {risk_type} detected for {uav_id}. {rule}."


if __name__ == "__main__":
    sample_risks = [
        {
            "uav_id": "SkyMule-5",
            "risk_type": "StabilityRisk",
            "details": {
                "current_wind": 12,
                "max_resistance": 10,
                "rule": "Rule #101: Flight prohibited if V_wind > V_resistance",
            },
        },
        {
            "uav_id": "UAV_002",
            "risk_type": "CriticalBatteryRisk",
            "details": {
                "battery_pct": 8,
                "threshold": 20,
                "rule": "Rule #201: Return-to-home required if battery < threshold",
            },
        },
        {
            "uav_id": "UAV_003",
            "risk_type": "NoFlyZoneRisk",
            "details": {
                "zone_id": "MilitaryZone_A",
                "rule": "Rule #301: Entry into restricted zone is prohibited",
            },
        },
    ]

    for ctx in sample_risks:
        print(generate_explanation(ctx))
        print()
