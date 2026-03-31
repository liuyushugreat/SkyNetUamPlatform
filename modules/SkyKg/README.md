# SkyKG: Neuro-Symbolic Knowledge Graph for UAM Risk Reasoning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Conference: KSEM 2026](https://img.shields.io/badge/Conference-KSEM_2026-green.svg)](https://ksem2026.rosc.org.cn/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **KSEM 2026 Reviewers:** For paper reproduction, go directly to **[`artifact_ksem2026/`](./artifact_ksem2026)** and run `bash run.sh`.

---

## Overview

**SkyKG** is a neuro-symbolic framework that integrates Knowledge Graphs with LLM-based reasoning for explainable risk assessment in Urban Air Mobility (UAM). It combines:

- **Structured Knowledge Representation** — A Low-Altitude Economy Ontology (RDF/OWL) encoding UAVs, airspace, regulations, and environmental conditions
- **Deterministic Compliance Checks** — SPARQL queries enforcing hard safety constraints
- **Semantic Risk Reasoning** — Retrieval-Augmented Generation (RAG) with DeepSeek-V3
- **Explainable Outputs** — Rule-grounded natural language explanations (RAR = 0.97)

---

## Repository Structure

```
modules/SkyKg/
├── README.md                           ← You are here
├── SkyNet_Knowledge_Engine/            # Core: ontology, reasoner, LLM agent
│   ├── ontology/skynet_core.ttl        #   RDF/OWL ontology
│   ├── reasoning/neuro_symbolic_reasoner.py  #   SPARQL retrieval + rule checking
│   └── llm_agent/                      #   DeepSeek API client + RAG explainer
├── voxel_airspace_core/                # Core: 3D voxelized airspace indexing
│   ├── adaptive_octree.py
│   ├── builder.py, indexer.py, manager.py
│   └── pathfinder.py
└── artifact_ksem2026/                  # Paper reproduction artifact
    ├── README.md                       #   Reviewer-facing guide
    ├── run.sh                          #   One-click reproduction
    ├── requirements.txt                #   Pinned dependencies
    ├── benchmark_comparison.py         #   Table 2: main benchmark
    ├── analyze_latency_tradeoff.py     #   Fig. 5: latency analysis
    ├── viz_ontology_structure.py       #   Fig. 2: ontology schema
    ├── viz_arch_placeholder.py         #   Fig. 1: architecture
    ├── generate_large_scale_dataset.py #   Generate N=1000 benchmark
    └── data/                           #   Pre-generated datasets
```

---

## Quick Start

```bash
git clone https://github.com/liuyushugreat/SkyNetUamPlatform.git
cd SkyNetUamPlatform/modules/SkyKg/artifact_ksem2026
export DEEPSEEK_API_KEY="your_key"
bash run.sh
```

See [`artifact_ksem2026/README.md`](./artifact_ksem2026/README.md) for detailed instructions.

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

Part of [SkyNetUamPlatform](https://github.com/liuyushugreat/SkyNetUamPlatform). Released under MIT License.
