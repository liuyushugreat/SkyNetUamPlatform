import logging
from rdflib import Graph, Namespace, Literal, RDF, XSD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BATTERY_CRITICAL_THRESHOLD = 20
BATTERY_LOW_THRESHOLD = 35


class SkyNetReasoner:
    """
    Neuro-Symbolic Reasoning Engine for the SkyKG framework.

    Uses RDF/OWL ontologies and SPARQL rules to infer risks from
    real-time UAV telemetry.  Supports three hard-rule categories:
      - StabilityRisk  (wind exceeds resistance)
      - BatteryRisk    (battery below safety threshold)
      - NoFlyZoneRisk  (UAV inside a restricted zone)
    """

    def __init__(self):
        self.graph = Graph()
        self.SKYNET = Namespace(
            "http://github.com/liuyushugreat/SkyNetUamPlatform/ontology#"
        )
        self.graph.bind("skynet", self.SKYNET)
        logger.info("SkyNetReasoner initialized.")

    def load_ontology(self, path: str):
        try:
            self.graph.parse(path, format="turtle")
            logger.info(f"Ontology loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load ontology: {e}")
            raise

    # ------------------------------------------------------------------
    # Data injection
    # ------------------------------------------------------------------

    def inject_realtime_data(self, uav_id: str, telemetry_data: dict):
        """Inject real-time UAV telemetry into the KG as RDF triples.

        Supported telemetry fields:
            wind_resistance (int), current_env_wind (int),
            battery (int), is_no_fly (bool), zone_id (str)
        """
        uav_uri = self.SKYNET[uav_id]
        self.graph.add((uav_uri, RDF.type, self.SKYNET.UAV))

        if "wind_resistance" in telemetry_data:
            self.graph.add((
                uav_uri,
                self.SKYNET.maxWindResistance,
                Literal(telemetry_data["wind_resistance"], datatype=XSD.integer),
            ))

        if "battery" in telemetry_data:
            self.graph.remove((uav_uri, self.SKYNET.currentBattery, None))
            self.graph.add((
                uav_uri,
                self.SKYNET.currentBattery,
                Literal(telemetry_data["battery"], datatype=XSD.integer),
            ))

        if "current_env_wind" in telemetry_data:
            self.graph.remove((uav_uri, self.SKYNET.currentEnvironmentWind, None))
            self.graph.add((
                uav_uri,
                self.SKYNET.currentEnvironmentWind,
                Literal(telemetry_data["current_env_wind"], datatype=XSD.integer),
            ))

        if telemetry_data.get("is_no_fly"):
            zone_id = telemetry_data.get("zone_id", "UnknownZone")
            zone_uri = self.SKYNET[zone_id]
            self.graph.add((zone_uri, RDF.type, self.SKYNET.RestrictedZone))
            self.graph.add((uav_uri, self.SKYNET.conflicts_with, zone_uri))

        logger.debug(f"Injected telemetry for {uav_id}")

    # ------------------------------------------------------------------
    # Multi-rule risk inference
    # ------------------------------------------------------------------

    _SPARQL_STABILITY = """
    PREFIX skynet: <http://github.com/liuyushugreat/SkyNetUamPlatform/ontology#>
    SELECT ?uav ?currWind ?maxWind WHERE {
        ?uav rdf:type skynet:UAV .
        ?uav skynet:maxWindResistance ?maxWind .
        ?uav skynet:currentEnvironmentWind ?currWind .
        FILTER (?currWind > ?maxWind)
    }
    """

    _SPARQL_BATTERY = """
    PREFIX skynet: <http://github.com/liuyushugreat/SkyNetUamPlatform/ontology#>
    SELECT ?uav ?bat WHERE {
        ?uav rdf:type skynet:UAV .
        ?uav skynet:currentBattery ?bat .
        FILTER (?bat < %d)
    }
    """ % BATTERY_LOW_THRESHOLD

    _SPARQL_NOFLY = """
    PREFIX skynet: <http://github.com/liuyushugreat/SkyNetUamPlatform/ontology#>
    SELECT ?uav ?zone WHERE {
        ?uav rdf:type skynet:UAV .
        ?uav skynet:conflicts_with ?zone .
        ?zone rdf:type skynet:RestrictedZone .
    }
    """

    def execute_risk_inference(self) -> list:
        """Run all hard-rule SPARQL queries and return detected risks.

        Returns:
            list[dict]: Each entry contains 'uav_id', 'risk_type', and
                        a 'details' dict with supporting evidence.
        """
        risks: list[dict] = []

        for row in self.graph.query(self._SPARQL_STABILITY):
            uid = str(row.uav).split("#")[-1]
            risks.append({
                "uav_id": uid,
                "risk_type": "StabilityRisk",
                "details": {
                    "current_wind": int(row.currWind),
                    "max_resistance": int(row.maxWind),
                    "rule": "Rule #101: Flight prohibited if V_wind > V_resistance",
                },
            })
            logger.warning(f"StabilityRisk: {uid} wind={row.currWind} > max={row.maxWind}")

        for row in self.graph.query(self._SPARQL_BATTERY):
            uid = str(row.uav).split("#")[-1]
            level = "CriticalBatteryRisk" if int(row.bat) < BATTERY_CRITICAL_THRESHOLD else "LowBatteryRisk"
            risks.append({
                "uav_id": uid,
                "risk_type": level,
                "details": {
                    "battery_pct": int(row.bat),
                    "threshold": BATTERY_CRITICAL_THRESHOLD if level.startswith("Critical") else BATTERY_LOW_THRESHOLD,
                    "rule": "Rule #201: Return-to-home required if battery < threshold",
                },
            })
            logger.warning(f"{level}: {uid} battery={row.bat}%")

        for row in self.graph.query(self._SPARQL_NOFLY):
            uid = str(row.uav).split("#")[-1]
            zid = str(row.zone).split("#")[-1]
            risks.append({
                "uav_id": uid,
                "risk_type": "NoFlyZoneRisk",
                "details": {
                    "zone_id": zid,
                    "rule": "Rule #301: Entry into restricted zone is prohibited",
                },
            })
            logger.warning(f"NoFlyZoneRisk: {uid} in zone {zid}")

        return risks

    def retrieve_context(self, uav_id: str) -> dict:
        """Retrieve all KG facts about a UAV for RAG prompt construction."""
        uav_uri = self.SKYNET[uav_id]
        ctx: dict = {"uav_id": uav_id, "properties": {}, "relations": []}
        for p, o in self.graph.predicate_objects(uav_uri):
            pname = str(p).split("#")[-1]
            if isinstance(o, Literal):
                ctx["properties"][pname] = o.toPython()
            else:
                ctx["relations"].append((pname, str(o).split("#")[-1]))
        return ctx


if __name__ == "__main__":
    import os

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ontology_path = os.path.join(base_path, "ontology", "skynet_core.ttl")

    reasoner = SkyNetReasoner()
    if os.path.exists(ontology_path):
        reasoner.load_ontology(ontology_path)

    reasoner.inject_realtime_data("UAV_001", {
        "wind_resistance": 5, "current_env_wind": 7, "battery": 80,
    })
    reasoner.inject_realtime_data("UAV_002", {
        "wind_resistance": 6, "current_env_wind": 4, "battery": 15,
    })
    reasoner.inject_realtime_data("UAV_003", {
        "wind_resistance": 8, "current_env_wind": 3, "battery": 60,
        "is_no_fly": True, "zone_id": "MilitaryZone_A",
    })

    for r in reasoner.execute_risk_inference():
        print(r)

