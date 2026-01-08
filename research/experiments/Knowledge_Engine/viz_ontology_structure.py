import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as mpatches

def generate_ontology_schema():
    # Setup paths
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent.parent
    
    # Output path
    output_dir = project_root / "research" / "papers" / "Knowledge_Engine"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "Fig_Ontology_Schema.png"

    # Create Graph
    G = nx.DiGraph()

    # Define Nodes with Categories
    core_classes = ["UAV", "Airspace", "Regulation", "Environment"]
    subclasses = ["RotaryWing", "FixedWing", "RestrictedZone", "FlightCorridor"]
    
    # Add nodes
    G.add_nodes_from(core_classes)
    G.add_nodes_from(subclasses)

    # Define Edges
    # Inheritance (Is-A)
    inheritance = [
        ("RotaryWing", "UAV"), ("FixedWing", "UAV"),
        ("RestrictedZone", "Airspace"), ("FlightCorridor", "Airspace")
    ]
    
    # Relations (Associations)
    relations = [
        ("UAV", "Airspace"), # located_in
        ("UAV", "RestrictedZone"), # conflicts_with
        ("UAV", "Regulation"), # governed_by
        ("RestrictedZone", "Regulation"), # enforces
        ("Environment", "Airspace") # affects
    ]

    G.add_edges_from(inheritance)
    G.add_edges_from(relations)

    # Layout
    # Custom positions to look like a hierarchy/schema
    pos = {
        "Environment": (0, 2),
        "Airspace": (1, 2),
        "RestrictedZone": (0.5, 3),
        "FlightCorridor": (1.5, 3),
        
        "UAV": (1, 1),
        "RotaryWing": (0.5, 0),
        "FixedWing": (1.5, 0),
        
        "Regulation": (2, 1.5)
    }

    # Plotting
    plt.figure(figsize=(10, 8), dpi=300)
    ax = plt.gca()
    
    # Draw Nodes
    # Core
    nx.draw_networkx_nodes(G, pos, nodelist=core_classes, node_size=3000, 
                           node_color='#D1E8FF', node_shape='s', edgecolors='#5D8AA8', linewidths=2)
    # Subclasses
    nx.draw_networkx_nodes(G, pos, nodelist=subclasses, node_size=2500, 
                           node_color='#F5F5F5', node_shape='o', edgecolors='#999999', linewidths=1)

    # Draw Edges
    # Inheritance (Solid)
    nx.draw_networkx_edges(G, pos, edgelist=inheritance, arrows=True, arrowstyle='-|>')
    
    # Relations (Dashed)
    nx.draw_networkx_edges(G, pos, edgelist=relations, style='dashed', 
                           edge_color='#555555', arrows=True, arrowstyle='-|>', connectionstyle="arc3,rad=0.1")

    # Labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif", font_weight="bold")

    # Edge Labels
    edge_labels = {
        ("UAV", "Airspace"): "located_in",
        ("UAV", "RestrictedZone"): "conflicts_with",
        ("UAV", "Regulation"): "governed_by",
        ("RestrictedZone", "Regulation"): "enforces",
        ("Environment", "Airspace"): "affects"
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # Legend
    blue_patch = mpatches.Patch(color='#D1E8FF', label='Core Concept')
    grey_patch = mpatches.Patch(color='#F5F5F5', label='Subclass')
    plt.legend(handles=[blue_patch, grey_patch], loc='upper right')

    plt.title("SkyKG Ontology Schema", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    
    plt.savefig(output_path)
    print(f"Ontology Schema generated at: {output_path}")

if __name__ == "__main__":
    generate_ontology_schema()

