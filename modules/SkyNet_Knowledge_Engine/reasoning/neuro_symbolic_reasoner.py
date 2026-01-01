import logging
from rdflib import Graph, Namespace, Literal, RDF, XSD
from rdflib.plugins.sparql import prepareQuery

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SkyNetReasoner:
    """
    SkyNetReasoner: A Neuro-Symbolic Reasoning Engine component.
    
    This class handles the 'Symbolic' part of the system, using RDF/OWL ontologies
    and SPARQL rules to infer risks and state changes based on real-time telemetry.
    """

    def __init__(self):
        """
        Initialize the Knowledge Graph and define Namespaces.
        """
        self.graph = Graph()
        self.SKYNET = Namespace("http://github.com/liuyushugreat/SkyNetUamPlatform/ontology#")
        self.graph.bind("skynet", self.SKYNET)
        logger.info("SkyNetReasoner initialized with empty graph.")

    def load_ontology(self, path: str):
        """
        Load the core SkyNet ontology from a Turtle (.ttl) file.
        
        Args:
            path (str): File path to the .ttl ontology file.
        """
        try:
            self.graph.parse(path, format="turtle")
            logger.info(f"Ontology loaded successfully from {path}")
        except Exception as e:
            logger.error(f"Failed to load ontology: {e}")
            raise

    def inject_realtime_data(self, uav_id: str, telemetry_data: dict):
        """
        Inject real-time UAV telemetry data into the Knowledge Graph as RDF triples.
        
        Args:
            uav_id (str): Unique identifier for the UAV (e.g., 'UAV_001').
            telemetry_data (dict): Dictionary containing data fields:
                - 'wind_resistance' (int): UAV's max wind resistance level.
                - 'current_env_wind' (int): Current detected wind speed level.
                - 'battery' (int): Current battery percentage.
        """
        # Define the UAV URI
        uav_uri = self.SKYNET[uav_id]

        # --- Symbolic Reasoning Logic: Data Injection ---
        
        # 1. Define Type: uav_uri is a skynet:UAV
        self.graph.add((uav_uri, RDF.type, self.SKYNET.UAV))

        # 2. Inject Static Property: maxWindResistance
        if 'wind_resistance' in telemetry_data:
            self.graph.add((
                uav_uri, 
                self.SKYNET.maxWindResistance, 
                Literal(telemetry_data['wind_resistance'], datatype=XSD.integer)
            ))

        # 3. Inject Dynamic Property: currentBattery
        if 'battery' in telemetry_data:
            # First remove old battery value if exists (to update state)
            self.graph.remove((uav_uri, self.SKYNET.currentBattery, None))
            self.graph.add((
                uav_uri, 
                self.SKYNET.currentBattery, 
                Literal(telemetry_data['battery'], datatype=XSD.integer)
            ))

        # 4. Inject Dynamic Environmental Data: currentEnvironmentWind
        # Note: This property represents the wind condition *experienced* by the UAV.
        if 'current_env_wind' in telemetry_data:
            # Remove old wind value
            self.graph.remove((uav_uri, self.SKYNET.currentEnvironmentWind, None))
            self.graph.add((
                uav_uri, 
                self.SKYNET.currentEnvironmentWind, 
                Literal(telemetry_data['current_env_wind'], datatype=XSD.integer)
            ))
            
        logger.debug(f"Injected telemetry for {uav_id}")

    def execute_risk_inference(self):
        """
        Execute SPARQL rules to detect logical conflicts and risks.
        
        Logic Target: Detect 'StabilityRisk' where currentEnvironmentWind > maxWindResistance.
        
        Returns:
            list: A list of tuples (uav_id, risk_type).
        """
        # --- Symbolic Reasoning Logic: Risk Inference ---
        
        query_str = """
        PREFIX skynet: <http://github.com/liuyushugreat/SkyNetUamPlatform/ontology#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        SELECT ?uav
        WHERE {
            ?uav rdf:type skynet:UAV .
            ?uav skynet:maxWindResistance ?maxWind .
            ?uav skynet:currentEnvironmentWind ?currWind .
            FILTER (?currWind > ?maxWind)
        }
        """
        
        risks = []
        try:
            results = self.graph.query(query_str)
            
            for row in results:
                # Extract the local name (e.g., 'UAV_001') from the full URI
                uav_uri = row.uav
                uav_id = uav_uri.split("#")[-1]
                
                risks.append((uav_id, "StabilityRisk"))
                logger.warning(f"Risk Detected: {uav_id} is facing StabilityRisk (Wind > MaxResistance).")
                
        except Exception as e:
            logger.error(f"Error executing risk inference: {e}")
            
        return risks

# Example usage for testing (if run directly)
if __name__ == "__main__":
    import os
    
    # Setup paths
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ontology_path = os.path.join(base_path, "ontology", "skynet_core.ttl")
    
    # Initialize Reasoner
    reasoner = SkyNetReasoner()
    
    # Load Ontology
    if os.path.exists(ontology_path):
        reasoner.load_ontology(ontology_path)
    else:
        logger.error(f"Ontology file not found at {ontology_path}")
    
    # Inject Data
    # UAV_001: Max Wind 5, Current Wind 7 -> Should trigger risk
    reasoner.inject_realtime_data("UAV_001", {
        "wind_resistance": 5,
        "current_env_wind": 7,
        "battery": 80
    })
    
    # UAV_002: Max Wind 6, Current Wind 4 -> Safe
    reasoner.inject_realtime_data("UAV_002", {
        "wind_resistance": 6,
        "current_env_wind": 4,
        "battery": 90
    })
    
    # Execute Reasoning
    detected_risks = reasoner.execute_risk_inference()
    print("Detected Risks:", detected_risks)

