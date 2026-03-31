import matplotlib.pyplot as plt
from pathlib import Path

def generate_arch_placeholder():
    # Setup paths
    current_dir = Path(__file__).resolve().parent
    output_dir = current_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "Fig_System_Arch.png"

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Draw simple blocks
    ax.add_patch(plt.Rectangle((0.1, 0.1), 0.8, 0.3, fill=True, color='lightgray', alpha=0.5))
    ax.text(0.5, 0.25, "Bottom Layer: Data Processing", ha='center', va='center', fontsize=14)
    
    ax.add_patch(plt.Rectangle((0.1, 0.5), 0.8, 0.4, fill=True, color='lightblue', alpha=0.5))
    ax.text(0.5, 0.7, "Top Layer: Neuro-Symbolic Reasoning", ha='center', va='center', fontsize=14)
    
    ax.arrow(0.5, 0.4, 0, 0.1, head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    plt.title("System Architecture (Placeholder)", fontsize=16)
    plt.axis('off')
    
    plt.savefig(output_path)
    print(f"Architecture Placeholder generated at: {output_path}")

if __name__ == "__main__":
    generate_arch_placeholder()

