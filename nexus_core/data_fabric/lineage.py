"""
SkyNet-RWA-Nexus Data Lineage Graph.

This module implements the Provenance Graph for all system data assets.
It adheres to the W3C PROV-O standard for provenance modeling.

Graph Structure:
    $G = (V, E)$
    Nodes $V$: {Entity, Activity, Agent}
    Edges $E$: {wasGeneratedBy, used, wasAttributedTo}

Use Case:
    Before minting a SkyRouteNFT, the system queries this graph to ensure
    the flight data node is connected to a trusted 'SimulationActivity' node.
"""

from enum import Enum
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
import uuid
import time

class NodeType(Enum):
    ENTITY = "Entity"       # e.g., FlightLog, VideoStream
    ACTIVITY = "Activity"   # e.g., SimulationRun, AUction
    AGENT = "Agent"         # e.g., UAV-001, Operator-A

class EdgeType(Enum):
    WAS_GENERATED_BY = "wasGeneratedBy" # Entity -> Activity
    USED = "used"                       # Activity -> Entity
    WAS_ATTRIBUTED_TO = "wasAttributedTo" # Entity -> Agent
    WAS_DERIVED_FROM = "wasDerivedFrom"   # Entity -> Entity

@dataclass
class LineageNode:
    id: str
    type: NodeType
    label: str
    metadata: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class LineageEdge:
    source_id: str
    target_id: str
    type: EdgeType
    timestamp: float = field(default_factory=time.time)

class DataLineageGraph:
    """
    In-memory implementation of the Data Lineage Graph.
    In production, this would wrap a Neo4j or ArangoDB driver.
    """
    def __init__(self):
        self.nodes: Dict[str, LineageNode] = {}
        self.edges: List[LineageEdge] = []
        # Adjacency list for fast traversal: source -> [(target, type)]
        self._adj: Dict[str, List[tuple]] = {} 

    def add_node(self, node: LineageNode):
        self.nodes[node.id] = node
        if node.id not in self._adj:
            self._adj[node.id] = []

    def add_edge(self, source_id: str, target_id: str, edge_type: EdgeType):
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError("Source or Target node does not exist.")
        
        edge = LineageEdge(source_id, target_id, edge_type)
        self.edges.append(edge)
        self._adj[source_id].append((target_id, edge_type))

    def trace_back(self, entity_id: str) -> List[LineageNode]:
        """
        Performs a BFS backward traversal to find the root cause/agent.
        Useful for auditing 'Who generated this data?'.
        """
        # Note: Since edges are directed (e.g., Entity -> Activity), 
        # standard trace usually follows the defined edges.
        # Here we implement a simple provenance scan.
        trace = []
        queue = [entity_id]
        visited = {entity_id}

        while queue:
            current = queue.pop(0)
            trace.append(self.nodes[current])
            
            if current in self._adj:
                for target, _ in self._adj[current]:
                    if target not in visited:
                        visited.add(target)
                        queue.append(target)
        
        return trace

    def verify_provenance(self, entity_id: str, required_agent_id: str) -> bool:
        """
        Oracle Check: Verify if an entity eventually points to a trusted agent.
        """
        trace = self.trace_back(entity_id)
        for node in trace:
            if node.type == NodeType.AGENT and node.id == required_agent_id:
                return True
        return False

