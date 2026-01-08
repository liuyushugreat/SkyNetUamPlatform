import json
import sys
import os
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np

# Path setup: Add project root to sys.path
# research/experiments/Knowledge_Engine -> project_root (4 levels up)
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

# Output Directory for Paper
PAPER_OUTPUT_DIR = project_root / "research" / "papers" / "Knowledge_Engine"
PAPER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables
load_dotenv(project_root / ".env")
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    print("WARNING: DEEPSEEK_API_KEY not found in environment variables or .env file.")

def get_llm():
    """Configure DeepSeek LLM"""
    if not api_key:
        # Return a mock or handle error appropriately if needed
        # For this script, we'll let it fail or use a placeholder if key missing
        return None
        
    return ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=api_key,
        openai_api_base="https://api.deepseek.com",
        temperature=0.1 # Low temperature for more deterministic reasoning
    )

def run_baseline_rule(case):
    """
    Method 1: Baseline Rule-Based System
    Simulates traditional flight control logic.
    Only checks:
    - No-fly zone violation
    - Low battery (mocked logic as data doesn't have battery explicitly in generator, using zone only here)
    
    LIMITATION: Ignores complex semantic checks like wind vs resistance.
    """
    zone = case.get("zone", {})
    
    # Simple hard rules
    if zone.get("is_no_fly"):
        return "High Risk"
    
    # Missing: Wind vs Resistance check
    
    return "Low Risk"

def run_baseline_direct_llm(case, llm):
    """
    Method 2: Direct LLM (Zero-Shot)
    Feeds raw JSON to LLM without specific domain knowledge context.
    """
    if not llm:
        return "Error"

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an aviation safety assistant. Analyze the input JSON and determine if the risk is 'High Risk' or 'Low Risk'. Output ONLY the label."),
        ("user", "Data: {data}")
    ])
    
    chain = prompt | llm
    
    try:
        response = chain.invoke({"data": json.dumps(case)})
        content = response.content.strip()
        if "High Risk" in content: return "High Risk"
        if "Low Risk" in content: return "Low Risk"
        return "Low Risk" # Default fallback
    except Exception as e:
        print(f"LLM Error: {e}")
        return "Error"

def run_proposed_skykg(case, llm):
    """
    Method 3: SkyKG (Neuro-Symbolic) - Ours
    Simulates:
    1. Retrieval of specific ontology rules (Symbolic)
    2. LLM reasoning with injected knowledge (Neural)
    """
    if not llm:
        return "Error"

    # Step 1: Simulated Knowledge Retrieval (In real system, this comes from Vector DB/SPARQL)
    # We retrieve relevant rules based on the context (e.g., wind data present -> retrieve wind rule)
    retrieved_rules = []
    
    if "wind_speed" in case.get("environment", {}) and "max_wind_resistance" in case.get("uav", {}):
        retrieved_rules.append("RULE: Stability Risk exists if environment.wind_speed > uav.max_wind_resistance.")
    
    if case.get("zone", {}).get("is_no_fly"):
        retrieved_rules.append("RULE: Airspace Violation if zone.is_no_fly is True.")

    rules_text = "\n".join(retrieved_rules)

    # Step 2: Neuro-Symbolic Reasoning
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are SkyKG, a neuro-symbolic reasoner. Use the retrieved Knowledge Graph rules to assess risk.\n"
                   "Knowledge:\n{rules}\n\n"
                   "Task: Analyze the flight data. If ANY rule is violated, return 'High Risk'. Otherwise 'Low Risk'. "
                   "Output ONLY the label."),
        ("user", "Flight Data: {data}")
    ])
    
    chain = prompt | llm
    
    try:
        response = chain.invoke({"rules": rules_text, "data": json.dumps(case)})
        content = response.content.strip()
        if "High Risk" in content: return "High Risk"
        if "Low Risk" in content: return "Low Risk"
        return "Low Risk"
    except Exception as e:
        print(f"LLM Error: {e}")
        return "Error"

def plot_confusion_matrix(y_true, y_pred, method_name):
    """
    Generate and save confusion matrix heatmap.
    """
    labels = sorted(list(set(y_true + y_pred)))
    # Ensure High Risk/Low Risk order if present
    if "High Risk" in labels and "Low Risk" in labels:
        labels = ["High Risk", "Low Risk"]
        
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, 
                yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix: {method_name}')
    
    output_path = PAPER_OUTPUT_DIR / f"Fig_Confusion_Matrix_{method_name.replace(' ', '_')}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")

def generate_latex_table(stats):
    """
    Generate LaTeX table code for performance metrics.
    stats: dict of method -> {metric: value}
    """
    latex_code = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Performance Comparison of Risk Detection Methods (N=1000)}",
        r"\label{tab:performance}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{Latency (ms)} \\ \midrule"
    ]
    
    for method, metrics in stats.items():
        row = f"{method} & {metrics['Accuracy']:.3f} & {metrics['Precision']:.3f} & {metrics['Recall']:.3f} & {metrics['F1']:.3f} & {metrics['Latency']:.1f} \\\\"
        latex_code.append(row)
        
    latex_code.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])
    
    # Save to file
    tex_path = PAPER_OUTPUT_DIR / "table_performance.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_code))
    
    print("\nLaTeX Table Code generated:")
    print("\n".join(latex_code))

def evaluate_benchmarks():
    dataset_path = current_dir / "ksem_large_dataset.json" # Updated dataset path
    if not dataset_path.exists():
        print("Dataset not found. Please run generate_large_scale_dataset.py first.")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print(f"Loaded {len(cases)} test cases.")
    llm = get_llm()
    if not llm:
        print("Skipping LLM-based methods due to missing API key.")
        return

    results = {
        "Baseline-Rule": [],
        "Baseline-LLM": [],
        "SkyKG-Ours": [],
        "Ground-Truth": []
    }
    
    latencies = {
        "Baseline-Rule": [],
        "Baseline-LLM": [],
        "SkyKG-Ours": []
    }

    print("Running evaluation...")
    for i, case in enumerate(cases):
        gt = case["ground_truth"]
        results["Ground-Truth"].append(gt)
        
        # 1. Baseline Rule
        start = time.time()
        pred_rule = run_baseline_rule(case)
        latencies["Baseline-Rule"].append((time.time() - start) * 1000)
        results["Baseline-Rule"].append(pred_rule)

        # 2. Baseline LLM
        start = time.time()
        pred_llm = run_baseline_direct_llm(case, llm)
        latencies["Baseline-LLM"].append((time.time() - start) * 1000)
        results["Baseline-LLM"].append(pred_llm)

        # 3. SkyKG
        start = time.time()
        pred_skykg = run_proposed_skykg(case, llm)
        latencies["SkyKG-Ours"].append((time.time() - start) * 1000)
        results["SkyKG-Ours"].append(pred_skykg)
            
        if (i+1) % 50 == 0:
            print(f"Processed {i+1}/{len(cases)} cases...")

    # Calculate Stats
    stats = {}
    methods = ["Baseline-Rule", "Baseline-LLM", "SkyKG-Ours"]
    
    for method in methods:
        y_true = results["Ground-Truth"]
        y_pred = results[method]
        
        # Plot Confusion Matrix
        plot_confusion_matrix(y_true, y_pred, method)
        
        # Calculate Metrics
        stats[method] = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, pos_label="High Risk", average='binary', zero_division=0),
            "Recall": recall_score(y_true, y_pred, pos_label="High Risk", average='binary', zero_division=0),
            "F1": f1_score(y_true, y_pred, pos_label="High Risk", average='binary', zero_division=0),
            "Latency": np.mean(latencies[method])
        }

    # Generate LaTeX Table
    generate_latex_table(stats)

if __name__ == "__main__":
    evaluate_benchmarks()

