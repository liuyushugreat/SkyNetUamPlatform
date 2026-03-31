# KSEM 2026 Artifact: SkyKG

**Paper:** *SkyKG: A Neuro-Symbolic Knowledge Graph Framework for Explainable Risk Reasoning in Urban Air Mobility*

**Authors:** Yushu Liu, Longbiao Wang, Chenglin Du, Haixiao Zhai

**Conference:** KSEM 2026 — 19th International Conference on Knowledge Science, Engineering and Management

---

## What This Directory Contains

This is the **self-contained reproduction artifact** for the KSEM 2026 paper. It lives inside the larger [SkyNetUamPlatform](https://github.com/liuyushugreat/SkyNetUamPlatform) repository but can be run independently.

| File / Directory | Maps to Paper | Description |
|-----------------|---------------|-------------|
| `benchmark_comparison.py` | **Table 2, Fig. 3, Fig. 4** | Main benchmark: Rule-Based vs Direct LLM vs SkyKG |
| `reproduce_table4.py` | **Table 4** | Explanation quality evaluation (RAR, LEC, UCR) |
| `analyze_latency_tradeoff.py` | **Fig. 5** | Inference latency distribution analysis |
| `experiment_runner.py` | — | Automated experiment pipeline (alternative entry) |
| `viz_ontology_structure.py` | **Fig. 2** | Ontology schema visualization |
| `viz_arch_placeholder.py` | **Fig. 1** | System architecture diagram |
| `generate_large_scale_dataset.py` | §4.1 | Generates 1,000-case balanced synthetic benchmark |
| `generate_ksem_dataset.py` | §4.1 | Generates 50-case quick-test set |
| `data/` | — | Pre-generated datasets (ready to use) |
| `run.sh` | — | One-click reproduction script |
| `requirements.txt` | — | Python dependencies |

### Core Modules (referenced by scripts above)

The experiment scripts import from these sibling modules in `modules/SkyKg/`:

- **`../SkyNet_Knowledge_Engine/`** — Ontology (RDF/OWL), SPARQL reasoner, DeepSeek LLM agent
- **`../voxel_airspace_core/`** — 3D voxelized airspace indexing

---

## Quick Start

### Prerequisites

- Python 3.10+
- DeepSeek API key (required for LLM-based methods)

### Option A: One-Click (Linux/macOS)

```bash
git clone https://github.com/liuyushugreat/SkyNetUamPlatform.git
cd SkyNetUamPlatform/modules/SkyKg/artifact_ksem2026

# Set your API key
export DEEPSEEK_API_KEY="your_key_here"

# Run everything
bash run.sh
```

### Option B: Step-by-Step

```bash
cd SkyNetUamPlatform/modules/SkyKg/artifact_ksem2026
pip install -r requirements.txt

# Table 2: Main benchmark (Rule vs LLM vs SkyKG)
python benchmark_comparison.py

# Table 4: Explanation quality (RAR, LEC, UCR)
python reproduce_table4.py

# Fig. 5: Latency analysis
python analyze_latency_tradeoff.py

# Fig. 2: Ontology schema
python viz_ontology_structure.py
```

### Output

All results are written to `outputs/` (auto-created):
- `outputs/Fig3_Accuracy_Comparison.png` — Per-scenario accuracy (Fig. 3)
- `outputs/Fig_Confusion_Matrix_*.png` — Confusion matrices (Fig. 4)
- `outputs/Fig_Latency_Analysis.png` — Latency distribution (Fig. 5)
- `outputs/Fig_Ontology_Schema.png` — Ontology schema (Fig. 2)
- Console output prints Table 2 metrics
- `outputs/table4_explanation_quality.json` — Per-case RAR/LEC/UCR scores (Table 4)

---

## Expected Results (Table 2, N=1000)

| Method | Accuracy | Precision | Recall | F1 | Latency (ms) |
|--------|:--------:|:---------:|:------:|:--:|:------------:|
| Baseline-Rule | 0.666 | 1.000 | 0.499 | 0.666 | ~0 |
| Baseline-LLM | 1.000 | 1.000 | 1.000 | 1.000 | ~1355 |
| **SkyKG (Ours)** | **1.000** | **1.000** | **1.000** | **1.000** | **~1363** |

> SkyKG's advantage over Direct LLM is in **explanation quality** (RAR=0.97 vs 0.31), not aggregate accuracy. See paper §4.6.

---

## Notes

- **API costs:** Running the full benchmark calls DeepSeek API ~2,000 times (1,000 cases x 2 LLM methods). Estimated cost: < $1 USD.
- **Runtime:** ~30-45 minutes depending on API response times.
- **Without API key:** The Rule-Based baseline will still run; LLM methods will be skipped with a warning.
