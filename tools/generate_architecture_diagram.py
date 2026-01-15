import matplotlib.pyplot as plt
import networkx as nx
import os

def generate_diagram_mpl():
    output_dir = r"D:\github_repos\SkyNetUamPlatform\research\papers\syspic"
    output_filename = "system_architecture.png"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    G = nx.DiGraph()

    # Define nodes with categories for coloring
    ui_nodes = {
        "CitizenApp": "Citizen App\n(Mobile/Web)",
        "OperatorApp": "Operator Ops\n(Dashboard)",
        "RegulatorApp": "Regulator App\n(Compliance)"
    }
    
    be_nodes = {
        "APIGateway": "API Gateway\n(NestJS)",
        "MissionService": "Mission Service\n(State Machine)"
    }
    
    kn_nodes = {
        "DeepSeek": "DeepSeek Agent\n(LLM)",
        "Reasoner": "Neuro-Symbolic\nReasoner",
        "Ontology": "Domain\nOntology",
        "Neo4j": "Neo4j\nGraph DB"
    }
    
    as_nodes = {
        "Octree": "Adaptive\nOctree",
        "Pathfinder": "4D Pathfinder",
        "AirspaceMgr": "Airspace\nManager"
    }
    
    rwa_nodes = {
        "Pricing": "Pricing\nEngine",
        "Tokenization": "Tokenization\nService",
        "DataFabric": "Data Fabric"
    }
    
    inf_nodes = {
        "SmartContracts": "Smart Contracts\n(Settlement)",
        "MAS": "MAS Sim\n(Training)"
    }

    # Add nodes to graph
    for k, v in {**ui_nodes, **be_nodes, **kn_nodes, **as_nodes, **rwa_nodes, **inf_nodes}.items():
        G.add_node(k, label=v)

    # Edges
    edges = [
        ("CitizenApp", "APIGateway"), ("OperatorApp", "APIGateway"), ("RegulatorApp", "APIGateway"),
        ("APIGateway", "MissionService"),
        ("MissionService", "Pathfinder"), ("MissionService", "Pricing"), ("MissionService", "DeepSeek"),
        ("DeepSeek", "Neo4j"), ("Reasoner", "Neo4j"), ("DeepSeek", "Ontology"),
        ("Pathfinder", "Octree"), ("AirspaceMgr", "Octree"),
        ("Pricing", "DataFabric"), ("Tokenization", "SmartContracts"), ("MissionService", "Tokenization"),
        ("MAS", "Pricing"), ("MAS", "AirspaceMgr")
    ]
    G.add_edges_from(edges)

    # Manual Layout Position (x, y)
    pos = {}
    
    # Layer 0 (Top) - UI
    pos["CitizenApp"] = (-3, 6)
    pos["OperatorApp"] = (0, 6)
    pos["RegulatorApp"] = (3, 6)

    # Layer 1 - Backend
    pos["APIGateway"] = (0, 4.5)
    pos["MissionService"] = (0, 3)

    # Layer 2 - Core Services (Three columns)
    # Left: Knowledge
    pos["DeepSeek"] = (-4, 2)
    pos["Ontology"] = (-5, 1)
    pos["Reasoner"] = (-3, 1)
    pos["Neo4j"] = (-4, 0)

    # Center: Airspace
    pos["Pathfinder"] = (0, 1.5)
    pos["AirspaceMgr"] = (-1, 0.5)
    pos["Octree"] = (0, -0.5)

    # Right: RWA
    pos["Pricing"] = (3, 2)
    pos["DataFabric"] = (4, 1)
    pos["Tokenization"] = (2, 1)
    
    # Layer 3 - Infrastructure
    pos["SmartContracts"] = (3, -0.5)
    pos["MAS"] = (0, -2)

    # Draw
    plt.figure(figsize=(16, 12))
    
    # Draw specific groups
    nx.draw_networkx_nodes(G, pos, nodelist=ui_nodes.keys(), node_color='#ADD8E6', node_shape='s', node_size=4000) # LightBlue
    nx.draw_networkx_nodes(G, pos, nodelist=be_nodes.keys(), node_color='#90EE90', node_shape='s', node_size=4000) # LightGreen
    nx.draw_networkx_nodes(G, pos, nodelist=kn_nodes.keys(), node_color='#FFD700', node_shape='o', node_size=3500) # Gold
    nx.draw_networkx_nodes(G, pos, nodelist=as_nodes.keys(), node_color='#DDA0DD', node_shape='o', node_size=3500) # Plum
    nx.draw_networkx_nodes(G, pos, nodelist=rwa_nodes.keys(), node_color='#E0FFFF', node_shape='o', node_size=3500) # LightCyan
    nx.draw_networkx_nodes(G, pos, nodelist=inf_nodes.keys(), node_color='#D3D3D3', node_shape='s', node_size=4000) # LightGray

    # Edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowstyle='-|>', arrowsize=20, width=1.5)

    # Labels
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')

    # Legend/Annotation
    plt.text(-6, 6, "Layer 1: User Interface", fontsize=12, fontweight='bold', color='gray')
    plt.text(-6, 4.5, "Layer 2: API & Orchestration", fontsize=12, fontweight='bold', color='gray')
    plt.text(-6, 2, "Layer 3: Core Engines", fontsize=12, fontweight='bold', color='gray')
    plt.text(-6, -2, "Layer 4: Simulation & Infra", fontsize=12, fontweight='bold', color='gray')

    plt.title("SkyNet UAM Platform Architecture", fontsize=18, pad=20)
    plt.axis('off')
    
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated: {output_path}")

if __name__ == "__main__":
    generate_diagram_mpl()
