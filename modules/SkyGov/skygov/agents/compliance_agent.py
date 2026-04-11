"""ComplianceAgent: deterministic SPARQL-based hard-rule checking with veto power."""

from __future__ import annotations

from typing import Any, Dict, List

from rdflib import Graph, Namespace, Literal, RDF, XSD

from .base_agent import BaseAgent, AgentResult, AgentVerdict, TraceEntry


SPARQL_QUERIES = {
    "wind_violation": {
        "id": "REG-WIND-001",
        "description": "Wind speed exceeds UAV max resistance",
        "query": """
            PREFIX skynet: <{ns}>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            SELECT ?uav ?currWind ?maxWind
            WHERE {{
                ?uav rdf:type skynet:UAV .
                ?uav skynet:maxWindResistance ?maxWind .
                ?uav skynet:currentEnvironmentWind ?currWind .
                FILTER (?currWind > ?maxWind)
            }}
        """,
    },
    "battery_critical": {
        "id": "REG-BAT-001",
        "description": "Battery level below emergency threshold (15%)",
        "query": """
            PREFIX skynet: <{ns}>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            SELECT ?uav ?battery
            WHERE {{
                ?uav rdf:type skynet:UAV .
                ?uav skynet:currentBattery ?battery .
                FILTER (?battery < 15)
            }}
        """,
    },
    "restricted_zone": {
        "id": "REG-ZONE-001",
        "description": "UAV entered restricted airspace",
        "query": """
            PREFIX skynet: <{ns}>
            SELECT ?uav ?zone
            WHERE {{
                ?uav rdf:type skynet:UAV .
                ?uav skynet:currentZone ?zone .
                ?zone rdf:type skynet:RestrictedZone .
            }}
        """,
    },
    "altitude_violation": {
        "id": "REG-ALT-001",
        "description": "Altitude exceeds maximum allowed for airspace",
        "query": """
            PREFIX skynet: <{ns}>
            SELECT ?uav ?alt ?maxAlt
            WHERE {{
                ?uav rdf:type skynet:UAV .
                ?uav skynet:currentAltitude ?alt .
                ?uav skynet:maxAllowedAltitude ?maxAlt .
                FILTER (?alt > ?maxAlt)
            }}
        """,
    },
    "visibility_low": {
        "id": "REG-VIS-001",
        "description": "Visibility below minimum requirement (1.5 km)",
        "query": """
            PREFIX skynet: <{ns}>
            SELECT ?uav ?vis
            WHERE {{
                ?uav rdf:type skynet:UAV .
                ?uav skynet:currentVisibility ?vis .
                FILTER (xsd:float(?vis) < 1.5)
            }}
        """,
    },
    "payload_overweight": {
        "id": "REG-LOAD-001",
        "description": "Payload exceeds maximum capacity",
        "query": """
            PREFIX skynet: <{ns}>
            SELECT ?uav ?payload ?maxPayload
            WHERE {{
                ?uav rdf:type skynet:UAV .
                ?uav skynet:currentPayload ?payload .
                ?uav skynet:maxPayloadCapacity ?maxPayload .
                FILTER (xsd:float(?payload) > xsd:float(?maxPayload))
            }}
        """,
    },
    "speed_violation": {
        "id": "REG-SPEED-001",
        "description": "Speed exceeds maximum allowed",
        "query": """
            PREFIX skynet: <{ns}>
            SELECT ?uav ?speed ?maxSpeed
            WHERE {{
                ?uav rdf:type skynet:UAV .
                ?uav skynet:currentSpeed ?speed .
                ?uav skynet:maxAllowedSpeed ?maxSpeed .
                FILTER (xsd:float(?speed) > xsd:float(?maxSpeed))
            }}
        """,
    },
    "temperature_extreme": {
        "id": "REG-TEMP-001",
        "description": "Temperature outside safe operating range (-20 to 45 C)",
        "query": """
            PREFIX skynet: <{ns}>
            SELECT ?uav ?temp
            WHERE {{
                ?uav rdf:type skynet:UAV .
                ?uav skynet:currentTemperature ?temp .
                FILTER (xsd:float(?temp) > 45 || xsd:float(?temp) < -20)
            }}
        """,
    },
}


class ComplianceAgent(BaseAgent):
    """Deterministic rule-checking agent with veto authority.

    Runs a battery of pre-defined SPARQL queries against the knowledge graph.
    Any match triggers a VIOLATION verdict that downstream agents cannot override.
    """

    name = "compliance"

    def __init__(self, config=None, ontology_path: str | None = None):
        super().__init__(config)
        self.graph = Graph()
        self.ns = "http://github.com/liuyushugreat/SkyNetUamPlatform/ontology#"
        self.SKYNET = Namespace(self.ns)
        self.graph.bind("skynet", self.SKYNET)
        if ontology_path:
            self.graph.parse(ontology_path, format="turtle")

    def inject_scenario(self, uav_id: str, data: Dict[str, Any]):
        """Inject a flight scenario into the local RDF graph for checking."""
        uav_uri = self.SKYNET[uav_id]
        self.graph.add((uav_uri, RDF.type, self.SKYNET.UAV))
        field_map = {
            "wind_resistance": ("maxWindResistance", XSD.integer),
            "current_env_wind": ("currentEnvironmentWind", XSD.integer),
            "battery": ("currentBattery", XSD.integer),
            "altitude": ("currentAltitude", XSD.integer),
            "max_altitude": ("maxAllowedAltitude", XSD.integer),
            "visibility_km": ("currentVisibility", XSD.float),
            "payload_kg": ("currentPayload", XSD.float),
            "max_payload_kg": ("maxPayloadCapacity", XSD.float),
            "speed_ms": ("currentSpeed", XSD.float),
            "max_speed_ms": ("maxAllowedSpeed", XSD.float),
            "temperature_c": ("currentTemperature", XSD.float),
        }
        for key, (pred_name, dtype) in field_map.items():
            if key in data:
                pred = self.SKYNET[pred_name]
                self.graph.remove((uav_uri, pred, None))
                self.graph.add((uav_uri, pred, Literal(data[key], datatype=dtype)))

    def _reset_graph(self):
        """Clear all instance data, keeping only ontology schema triples."""
        self.graph = Graph()
        self.graph.bind("skynet", self.SKYNET)

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        self._reset_graph()
        uav_id = context.get("uav_id", "UNKNOWN")
        telemetry = context.get("telemetry", {})
        self.inject_scenario(uav_id, telemetry)

        violations: List[TraceEntry] = []
        for rule_key, rule_def in SPARQL_QUERIES.items():
            query = rule_def["query"].format(ns=self.ns)
            try:
                results = list(self.graph.query(query))
                if results:
                    violations.append(
                        TraceEntry(
                            step=f"sparql_{rule_key}",
                            source="compliance_agent",
                            rule_ids=[rule_def["id"]],
                            detail=f"{rule_def['description']} — {len(results)} match(es)",
                        )
                    )
            except Exception as e:
                self.logger.error("SPARQL query %s failed: %s", rule_key, e)

        if violations:
            return AgentResult(
                agent_name=self.name,
                verdict=AgentVerdict.VIOLATION,
                confidence=1.0,
                payload={"violation_count": len(violations)},
                traces=violations,
            )

        return AgentResult(
            agent_name=self.name,
            verdict=AgentVerdict.SAFE,
            confidence=1.0,
            traces=[
                TraceEntry(
                    step="sparql_check_all",
                    source="compliance_agent",
                    detail=f"All {len(SPARQL_QUERIES)} rules passed",
                )
            ],
        )
