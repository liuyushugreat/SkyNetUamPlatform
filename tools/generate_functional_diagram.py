import matplotlib.pyplot as plt
import networkx as nx
import os

def generate_functional_diagram(lang='en'):
    output_dir = r"D:\github_repos\SkyNetUamPlatform\research\papers\syspic"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Configure fonts
    if lang == 'cn':
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
        output_filename = "functional_structure_cn.png"
        root_label = "SkyNet UAM 平台"
        
        # Labels map for Chinese
        labels = {
            "Root": "SkyNet UAM 平台",
            "UI": "用户交互层",
            "Engines": "核心引擎层",
            "Infra": "基础设施与服务层",
            
            "CitizenApp": "市民端 App\n(预订/追踪)",
            "OperatorApp": "运营商端\n(机队/监控)",
            "RegulatorApp": "监管端\n(合规/审计)",
            
            "Knowledge": "知识引擎\n(Knowledge Engine)",
            "Airspace": "体素空域核心\n(Voxel Core)",
            "RWA": "RWA 核心\n(资产定价)",
            
            "Nexus": "Nexus Core\n(资产化/结算)",
            "MAS": "多智能体仿真\n(MAS)",
            "Backend": "后端服务\n(API/任务)",
            
            # Functions
            "K_Func": "· 神经符号推理\n· 知识图谱 (Neo4j)\n· LLM Agent (DeepSeek)",
            "A_Func": "· 自适应八叉树\n· 4D 路径规划\n· 空域管理",
            "R_Func": "· 神经定价引擎\n· 拓扑度量\n· 估值模型",
            "N_Func": "· 资产代币化\n· 智能合约结算\n· 联邦学习",
            "M_Func": "· MADDPG 训练\n· 市场仿真\n· 冲突模拟",
            "B_Func": "· 任务状态机\n· 身份认证\n· API 网关"
        }
    else:
        output_filename = "functional_structure.png"
        root_label = "SkyNet UAM Platform"
        
        # Labels map for English
        labels = {
            "Root": "SkyNet UAM Platform",
            "UI": "User Interaction Layer",
            "Engines": "Core Engines Layer",
            "Infra": "Infrastructure & Services",
            
            "CitizenApp": "Citizen App\n(Booking/Tracking)",
            "OperatorApp": "Operator Ops\n(Fleet/Monitor)",
            "RegulatorApp": "Regulator Portal\n(Compliance)",
            
            "Knowledge": "Knowledge Engine",
            "Airspace": "Voxel Airspace Core",
            "RWA": "RWA Core",
            
            "Nexus": "Nexus Core\n(Assetization)",
            "MAS": "MAS Simulation",
            "Backend": "Backend Services",
            
            # Functions
            "K_Func": "· Neuro-Symbolic Reasoning\n· Knowledge Graph (Neo4j)\n· LLM Agent (DeepSeek)",
            "A_Func": "· Adaptive Octree\n· 4D Pathfinding\n· Airspace Manager",
            "R_Func": "· Neural Pricing Engine\n· Topology Metrics\n· Valuation Models",
            "N_Func": "· Asset Tokenization\n· Contract Settlement\n· Federated Learning",
            "M_Func": "· MADDPG Training\n· Market Sim\n· Conflict Sim",
            "B_Func": "· Mission State Machine\n· Auth Service\n· API Gateway"
        }

    G = nx.DiGraph()
    
    # Define Hierarchy
    structure = {
        "Root": ["UI", "Engines", "Infra"],
        "UI": ["CitizenApp", "OperatorApp", "RegulatorApp"],
        "Engines": ["Knowledge", "Airspace", "RWA"],
        "Infra": ["Backend", "Nexus", "MAS"]
    }
    
    # Function nodes mapping (Parent -> Function Node)
    func_map = {
        "Knowledge": "K_Func",
        "Airspace": "A_Func",
        "RWA": "R_Func",
        "Nexus": "N_Func",
        "MAS": "M_Func",
        "Backend": "B_Func"
    }

    # Add nodes and edges
    for parent, children in structure.items():
        G.add_node(parent, label=labels[parent])
        for child in children:
            G.add_node(child, label=labels[child])
            G.add_edge(parent, child)
            
            # Add function node if exists
            if child in func_map:
                f_node = func_map[child]
                G.add_node(f_node, label=labels[f_node])
                G.add_edge(child, f_node)

    # Layout calculation
    pos = {}
    
    # Root
    pos["Root"] = (0, 10)
    
    # Level 1
    pos["UI"] = (-6, 8)
    pos["Engines"] = (0, 8)
    pos["Infra"] = (6, 8)
    
    # Level 2 (UI)
    pos["CitizenApp"] = (-8, 6)
    pos["OperatorApp"] = (-6, 6)
    pos["RegulatorApp"] = (-4, 6)
    
    # Level 2 (Engines)
    pos["Knowledge"] = (-2, 6)
    pos["Airspace"] = (0, 6)
    pos["RWA"] = (2, 6)
    
    # Level 2 (Infra)
    pos["Backend"] = (4, 6)
    pos["Nexus"] = (6, 6)
    pos["MAS"] = (8, 6)
    
    # Level 3 (Functions) - Shifted down
    pos["K_Func"] = (-2, 4)
    pos["A_Func"] = (0, 4)
    pos["R_Func"] = (2, 4)
    
    pos["B_Func"] = (4, 4)
    pos["N_Func"] = (6, 4)
    pos["M_Func"] = (8, 4)

    # Draw
    plt.figure(figsize=(20, 12))
    
    # Draw Nodes
    # Root
    nx.draw_networkx_nodes(G, pos, nodelist=["Root"], node_color='#4682B4', node_shape='s', node_size=6000)
    # Level 1
    nx.draw_networkx_nodes(G, pos, nodelist=["UI", "Engines", "Infra"], node_color='#87CEEB', node_shape='s', node_size=5000)
    # Level 2 (Modules)
    module_nodes = ["CitizenApp", "OperatorApp", "RegulatorApp", "Knowledge", "Airspace", "RWA", "Backend", "Nexus", "MAS"]
    nx.draw_networkx_nodes(G, pos, nodelist=module_nodes, node_color='#98FB98', node_shape='o', node_size=4000)
    # Level 3 (Functions)
    func_nodes = list(func_map.values())
    nx.draw_networkx_nodes(G, pos, nodelist=func_nodes, node_color='#FFD700', node_shape='s', node_size=4500)

    # Draw Edges
    # Structure edges
    structure_edges = []
    for p, children in structure.items():
        for c in children:
            structure_edges.append((p, c))
    nx.draw_networkx_edges(G, pos, edgelist=structure_edges, edge_color='gray', arrows=False, width=2)
    
    # Function edges
    func_edges = []
    for p, c in func_map.items():
        func_edges.append((p, c))
    nx.draw_networkx_edges(G, pos, edgelist=func_edges, edge_color='gray', style='dashed', arrows=True, width=1.5)

    # Labels
    # Extract labels for current graph nodes
    node_labels = {n: labels[n] for n in G.nodes() if n in labels}
    nx.draw_networkx_labels(G, pos, node_labels, font_size=9, font_weight='bold')

    plt.title(f"{labels['Root']} - Functional Structure", fontsize=16, pad=20)
    plt.axis('off')
    
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated: {output_path}")

if __name__ == "__main__":
    generate_functional_diagram('en')
    generate_functional_diagram('cn')
