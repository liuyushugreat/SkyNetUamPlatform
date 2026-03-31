# SkyKG: A Neuro-Symbolic Knowledge Graph Framework for Explainable Risk Reasoning in Urban Air Mobility

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Conference: KSEM 2026](https://img.shields.io/badge/Conference-KSEM_2026-green.svg)](https://ksem2026.rosc.org.cn/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Official implementation** of SkyKG, a neuro-symbolic framework that integrates Knowledge Graphs with LLM-based reasoning for explainable risk assessment in Urban Air Mobility (UAM).

---

## Overview

SkyKG addresses the gap between rigid rule-based aviation safety systems and opaque data-driven models by combining:

- **Structured Knowledge Representation**: A Low-Altitude Economy Ontology encoding UAVs, airspace, regulations, and environmental conditions as RDF triples
- **Deterministic Compliance Checks**: SPARQL queries that enforce hard safety constraints
- **Semantic Risk Reasoning**: Retrieval-Augmented Generation (RAG) with DeepSeek-V3 for context-dependent risk assessment
- **Explainable Outputs**: Natural language explanations grounded in retrieved ontological rules

---

## Repository Structure

```
modules/SkyKg/
├── SkyNet_Knowledge_Engine/         # Core SkyKG implementation
│   ├── ontology/
│   │   └── skynet_core.ttl          # RDF/OWL ontology (UAVs, airspace, risks)
│   ├── reasoning/
│   │   └── neuro_symbolic_reasoner.py  # SPARQL retrieval + rule checking
│   ├── llm_agent/
│   │   ├── deepseek_client.py       # DeepSeek API integration
│   │   ├── risk_explainer.py        # RAG-based explanation generation
│   │   └── test_deepseek.py         # API connectivity test
│   └── README.md
├── voxel_airspace_core/             # 3D voxelized airspace indexing
│   ├── adaptive_octree.py
│   ├── builder.py
│   ├── indexer.py
│   ├── manager.py
│   └── pathfinder.py
├── experiments/                     # Paper experiment scripts & data
│   ├── benchmark_comparison.py      # Main benchmark: Rule vs LLM vs SkyKG
│   ├── analyze_latency_tradeoff.py  # Latency analysis (Fig. 5 in paper)
│   ├── experiment_runner.py         # Automated experiment pipeline
│   ├── generate_ksem_dataset.py     # Generate 50-sample test set
│   ├── generate_large_scale_dataset.py  # Generate 1000-sample benchmark
│   ├── viz_ontology_structure.py    # Generate ontology schema figure
│   ├── viz_arch_placeholder.py      # Generate architecture figure
│   └── data/
│       ├── ksem_large_dataset.json  # 1,000 synthetic test cases (N=1000)
│       └── ksem_test_cases.json     # 50-sample quick-test set
└── README.md                        # This file
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- DeepSeek API key (set as `DEEPSEEK_API_KEY` in `.env` at project root)

### Installation

```bash
git clone https://github.com/liuyushugreat/SkyNetUamPlatform.git
cd SkyNetUamPlatform

# Install dependencies
pip install langchain-openai python-dotenv scikit-learn seaborn matplotlib networkx

# Set your DeepSeek API key
echo "DEEPSEEK_API_KEY=your_key_here" > .env
```

### Run the Main Benchmark (Table 2 in paper)

```bash
cd modules/SkyKg/experiments
python benchmark_comparison.py
```

This runs all three methods (Rule-Based, Direct LLM, SkyKG) on the 1,000-case synthetic dataset and outputs:
- Accuracy, Precision, Recall, F1-Score for each method
- Confusion matrices
- Per-scenario accuracy comparison chart
- Latency measurements

### Generate Paper Figures

```bash
python viz_ontology_structure.py   # Fig. 2: Ontology schema
python viz_arch_placeholder.py     # Fig. 1: System architecture
python analyze_latency_tradeoff.py # Fig. 5: Latency distribution
```

### Regenerate Synthetic Dataset

```bash
python generate_large_scale_dataset.py  # 1,000 balanced test cases
python generate_ksem_dataset.py         # 50-sample quick test
```

---

## Key Results (Table 2 in paper, N=1000)

| Method | Accuracy | Precision | Recall | F1-Score | Latency (ms) |
|--------|:--------:|:---------:|:------:|:--------:|:------------:|
| Baseline-Rule | 0.666 | 1.000 | 0.499 | 0.666 | ~0 |
| Baseline-LLM | 1.000 | 1.000 | 1.000 | 1.000 | 1355.2 |
| **SkyKG (Ours)** | **1.000** | **1.000** | **1.000** | **1.000** | **1362.7** |

> **Note**: Both LLM-based methods achieve identical aggregate metrics on this controlled synthetic benchmark. SkyKG's core advantage lies in **explanation quality**: Rule Alignment Rate = 0.97 vs. 0.31 for the Direct LLM baseline (see paper Section 4.6).

---

## Citation

```bibtex
@inproceedings{liu2026skykg,
  title     = {SkyKG: A Neuro-Symbolic Knowledge Graph Framework for
               Explainable Risk Reasoning in Urban Air Mobility},
  author    = {Liu, Yushu and Wang, Longbiao and Du, Chenglin and Zhai, Haixiao},
  booktitle = {Proceedings of the 19th International Conference on
               Knowledge Science, Engineering and Management (KSEM)},
  year      = {2026}
}
```

## License

This project is part of [SkyNetUamPlatform](https://github.com/liuyushugreat/SkyNetUamPlatform) and is released under the MIT License.
