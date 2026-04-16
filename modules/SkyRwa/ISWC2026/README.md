# ISWC 2026 Artifact: SkyRwa

**Paper:** *Modeling Governable Flight-to-Asset Lifecycles with Knowledge Graphs, SHACL, and Provenance*

**Conference:** ISWC 2026 — 25th International Semantic Web Conference

---

## What This Directory Contains

This is the **self-contained reproduction artifact** for the ISWC 2026 paper. It lives inside the larger [SkyNetUamPlatform](https://github.com/liuyushugreat/SkyNetUamPlatform) repository but can be run independently.

| File | Maps to Paper | Description |
|------|---------------|-------------|
| `reproduce_table5.py` | **Table 5** (§7.1) | Benchmark dataset: 105 flights across 10 scenarios |
| `reproduce_table6.py` | **Table 6** (§7.2) | Baseline comparison: JSON-scan vs SPARQL (4 query tasks) |
| `reproduce_semantic_baseline.py` | **Table 7** (§7.3) | Semantic baseline: Lifecycle KG vs Flat KG (4 audit tasks) |
| `reproduce_table7.py` | **Table 8** (§7.4) | Governance ablation: Python vs SHACL vs Combined (6 violation types) |
| `reproduce_table8.py` | **Table 9** (§7.5) | Scalability: overhead for 5–1000 flights |
| `reproduce_table9.py` | **Table 10** (§7.7) | SPARQL competency questions (CQ1–CQ6) + analytical queries |
| `reproduce_robustness.py` | **§7.6** | Robustness: multi-run stability, scale sensitivity, threshold sweep |
| `reproduce_case_studies.py` | **§7.8** | Four case studies: promotion, failure, audit, explainability |
| `reproduce_ontology_quality.py` | **Table 4** (§4.5) | Ontology quality: OOPS!-style pitfall scan, OWL DL consistency, CQ→construct mapping |
| `reproduce_user_study.py` | **Table 11** (§7.9) | Pilot expert evaluation: 4 experts × 4 tasks × 3 interfaces |
| `reproduce_validation.py` | **§5** | SHACL + SHACL-SPARQL governance validation coverage |
| `run.sh` | — | One-click reproduction (Linux/macOS) |
| `run.ps1` | — | One-click reproduction (Windows) |
| `requirements.txt` | — | Python dependencies |
| `data/` | — | Generated benchmark data (auto-created) |
| `outputs/` | — | Experiment result JSONs (auto-created) |

### Core Modules (referenced by scripts above)

The experiment scripts import from these sibling directories in `modules/SkyRwa/`:

- **`ontology/`** — Domain ontology (13 classes, 26 properties), PROV-O/DCAT/ODRL alignments
- **`rdf/`** — RDF mapper, serializer, graph store, SPARQL execution
- **`shapes/`** — 5 SHACL constraint shapes
- **`queries/`** — 6 competency + 4 analytical SPARQL queries
- **`semantic_rules/`** — SHACL validator, SPARQL governance rules, promotion rules, explanation builder
- **`productization/`** — Multi-flight aggregation and product catalogue
- **`provenance/`** — Ed25519 signing and evidence builder
- **`valuation/`** — Rule-based + product-level valuation with explanations
- **`benchmarks/`** — Benchmark generator (105 flights, 10 scenarios)

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- No external API keys required — all experiments run locally
- No GPU required

### Option A: One-Click (Linux/macOS)

```bash
git clone https://github.com/liuyushugreat/SkyNetUamPlatform.git
cd SkyNetUamPlatform/modules/SkyRwa/ISWC2026

bash run.sh
```

### Option B: One-Click (Windows PowerShell)

```powershell
git clone https://github.com/liuyushugreat/SkyNetUamPlatform.git
cd SkyNetUamPlatform\modules\SkyRwa\ISWC2026

.\run.ps1
```

### Option C: Step-by-Step

```bash
cd SkyNetUamPlatform/modules/SkyRwa/ISWC2026
pip install -r requirements.txt

# Table 5: Generate 105-flight benchmark
python reproduce_table5.py

# Table 6: JSON-scan vs SPARQL baseline
python reproduce_table6.py

# Table 7: Semantic baseline (Lifecycle KG vs Flat KG)
python reproduce_semantic_baseline.py

# Table 8: Governance ablation (Python vs SHACL vs Combined)
python reproduce_table7.py

# Table 9: Scalability (5–1000 flights)
python reproduce_table8.py

# Table 10: SPARQL competency questions (CQ1–CQ6)
python reproduce_table9.py

# Section 7.6: Robustness (multi-run stability, scale, thresholds)
python reproduce_robustness.py

# Section 7.8: Case studies
python reproduce_case_studies.py

# Section 4.5: Ontology quality assessment (OOPS!-style pitfalls, consistency, CQ mapping)
python reproduce_ontology_quality.py

# Table 11 / Section 7.9: Pilot expert evaluation
python reproduce_user_study.py

# Section 5: SHACL + governance validation coverage
python reproduce_validation.py
```

### Output

- Console prints reproduce the tables from the paper
- JSON results are saved to `outputs/` (auto-created)
- Benchmark data is saved to `data/` (auto-created)

---

## Expected Results

### Table 5: Benchmark (§7.1)

| Scenario | Flights | Tradable | Governance Path |
|----------|:-------:|:--------:|-----------------|
| Clean route survey | 12 | 12 | Direct promotion |
| Night flight | 8 | 8 | Standard governance |
| Weather disturbance | 10 | 10 | Desensitization gate |
| Near-NFZ event | 8 | 3 | Mixed: pass / non-transfer |
| Anomaly maintenance | 10 | 0 | Standard governance |
| Emergency logistics | 8 | 0 | Mission failure path |
| Low-quality / incomplete | 12 | 0 | Quality failure |
| Rights-conflicted | 8 | 0 | Aggregation edge case |
| Beyond-VLOS operations | 15 | 12 | Range/link edge cases |
| Urban corridor multi-stop | 14 | 0 | Urban density / NFZ |
| **Total** | **105** | **45** | 10 distinct paths |

### Table 6: Baseline Comparison (§7.2)

| Task | J-LoC | S-LoC | JSON (ms) | SPARQL (ms) |
|------|:-----:|:-----:|:---------:|:-----------:|
| Tradable assets | 6 | 8 | <1 | ~30 |
| Revenue by participant | 8 | 8 | <1 | ~3 |
| Governance violations | 6 | 9 | <1 | ~36 |
| Product lineage (3-hop) | 10+ | 8 | <1 | ~2 |

### Table 8: Governance Ablation (§7.4)

| ID | Violation Type | Python | SHACL | Combined |
|----|----------------|:------:|:-----:|:--------:|
| V1 | Missing evidence digest | – | YES | YES |
| V2 | Missing derivation link | – | YES | YES |
| V3 | Low compliance + tradable | YES | – | YES |
| V4 | High risk + tradable | YES | – | YES |
| V5 | Missing rights on tradable | – | YES | YES |
| V6 | Incomplete mission + tradable | YES | – | YES |
| | **Detection rate** | **50%** | **50%** | **100%** |

### Table 9: Scalability (§7.5)

| N | Pipeline (ms) | RDF Map (ms) | Serialize (ms) | SHACL (ms) | Triples | T/flight |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 5 | ~0.7 | ~2.8 | ~5.7 | ~78 | ~336 | ~67 |
| 100 | ~6.1 | ~48.6 | ~94.4 | ~397 | ~6606 | ~66 |
| 1000 | ~58.2 | ~489.5 | ~971.1 | ~9634 | ~66006 | ~66 |

### Table 10: Competency Questions (§7.7)

All 6 CQs return correct results. CQ3 and CQ6 demonstrate multi-hop graph traversal.

---

## Runtime

- **Total runtime:** ~2–5 minutes (depends on hardware)
- **No network access required** — all experiments are self-contained
- **Disk usage:** ~20 MB for generated data and outputs

---

## Running Tests

The full test suite (13 test modules) can be run from the modules directory:

```bash
cd SkyNetUamPlatform/modules
python -m pytest SkyRwa/tests/ -v
```

---

## Directory Structure (Full Module)

```
modules/SkyRwa/
├── ISWC2026/                  ← YOU ARE HERE (reproduction artifact)
│   ├── README.md
│   ├── run.sh / run.ps1       # One-click reproduction
│   ├── requirements.txt       # Pinned dependencies
│   ├── reproduce_table5.py    # Table 5 (§7.1): benchmark dataset
│   ├── reproduce_table6.py    # Table 6 (§7.2): JSON vs SPARQL
│   ├── reproduce_semantic_baseline.py  # Table 7 (§7.3): Lifecycle KG vs Flat KG
│   ├── reproduce_table7.py    # Table 8 (§7.4): governance ablation
│   ├── reproduce_table8.py    # Table 9 (§7.5): scalability
│   ├── reproduce_table9.py    # Table 10 (§7.7): competency questions
│   ├── reproduce_robustness.py    # §7.6: robustness study
│   ├── reproduce_case_studies.py  # §7.8: case studies
│   ├── reproduce_user_study.py    # Table 11 (§7.9): pilot expert evaluation
│   ├── reproduce_ontology_quality.py  # Table 4 (§4.5): OOPS!-style pitfalls + CQ map
│   ├── reproduce_validation.py    # §5: validation coverage
│   ├── data/                  # Generated benchmark data
│   └── outputs/               # Experiment result JSONs
│
├── ontology/                  # Domain ontology (Turtle)
│   ├── skyrwa.ttl             # 13 classes, 26 properties
│   ├── alignments.ttl         # PROV-O / DCAT / ODRL / Schema.org
│   └── prefixes.ttl           # Shared namespace prefixes
├── shapes/                    # 5 SHACL constraint shapes
├── queries/                   # 10 SPARQL queries (6 CQ + 4 analytical)
├── rdf/                       # RDF serialization layer
├── semantic_rules/            # Governance / promotion / explanation rules
├── productization/            # Multi-flight aggregation
├── provenance/                # Ed25519 signing + evidence builder
├── valuation/                 # Rule-based + product valuation
├── benchmarks/                # Benchmark generator
├── experiments/               # Evaluation scripts (called by ISWC2026/)
├── models/                    # Pydantic data models
├── pipeline/                  # End-to-end orchestrator
├── ingest/                    # Flight data ingestion
├── rights/                    # Governance engine
├── settlement/                # Revenue recording & settlement
├── storage/                   # JSON file store
├── tests/                     # 13 pytest test modules
└── README.md                  # Module overview
```
