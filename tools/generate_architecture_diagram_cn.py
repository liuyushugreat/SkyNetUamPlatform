import matplotlib.pyplot as plt
import networkx as nx
import os

def generate_diagram_mpl_cn():
    # Set Chinese font
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    output_dir = r"D:\github_repos\SkyNetUamPlatform\research\papers\syspic"
    output_filename = "system_architecture_cn.png"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    G = nx.DiGraph()

    # Define nodes with categories for coloring
    ui_nodes = {
        "CitizenApp": "市民端 App\n(移动/Web)",
        "OperatorApp": "运营商端\n(仪表盘)",
        "RegulatorApp": "监管端 App\n(合规审查)"
    }
    
    be_nodes = {
        "APIGateway": "API 网关\n(NestJS)",
        "MissionService": "任务服务\n(状态机)"
    }
    
    kn_nodes = {
        "DeepSeek": "DeepSeek Agent\n(大语言模型)",
        "Reasoner": "神经符号\n推理机",
        "Ontology": "领域\n本体",
        "Neo4j": "Neo4j\n图数据库"
    }
    
    as_nodes = {
        "Octree": "自适应\n八叉树",
        "Pathfinder": "4D 路径规划",
        "AirspaceMgr": "空域\n管理器"
    }
    
    rwa_nodes = {
        "Pricing": "定价\n引擎",
        "Tokenization": "通证化\n服务",
        "DataFabric": "数据编织"
    }
    
    inf_nodes = {
        "SmartContracts": "智能合约\n(结算)",
        "MAS": "多智能体仿真\n(训练环境)"
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
    plt.text(-6, 6, "第 1 层：用户界面", fontsize=12, fontweight='bold', color='gray')
    plt.text(-6, 4.5, "第 2 层：API 与编排", fontsize=12, fontweight='bold', color='gray')
    plt.text(-6, 2, "第 3 层：核心引擎", fontsize=12, fontweight='bold', color='gray')
    plt.text(-6, -2, "第 4 层：仿真与基础设施", fontsize=12, fontweight='bold', color='gray')

    plt.title("SkyNet UAM 平台架构图", fontsize=18, pad=20)
    plt.axis('off')
    
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated: {output_path}")

if __name__ == "__main__":
    generate_diagram_mpl_cn()
