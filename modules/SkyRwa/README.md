# SkyRwa: Modeling Governable Flight-to-Asset Lifecycles with Knowledge Graphs, SHACL, and Provenance

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Conference: ISWC 2026](https://img.shields.io/badge/Conference-ISWC_2026-green.svg)](https://iswc2026.semanticweb.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **ISWC 2026 Reviewers:** For paper reproduction, go directly to **[`ISWC2026/`](./ISWC2026)** and run `bash run.sh` (Linux/macOS) or `.\run.ps1` (Windows).

---

## Overview

**SkyRwa** formalizes the governance transitions that transform raw UAM flight data into tradable data assets as first-class semantic objects in a knowledge graph. The core model is a **four-tier governance lifecycle**:

- **Tier 1: Flight Evidence** — Raw attestation with SHA-256 digest + Ed25519 signature. NOT tradable.
- **Tier 2: Asset Candidate** — Governed evidence with valuation and rights profile. Individually queryable.
- **Tier 3: Governed Data Product** — Aggregated from multiple candidates. Catalogued and licensable.
- **Tier 4: Revenue Right** — Downstream entitlement for revenue distribution.

The system is grounded in a **domain ontology (13 classes, 26 properties)** aligned with PROV-O, DCAT, ODRL 2.2, and Schema.org, validated through **5 SHACL shapes + SHACL-SPARQL constraints**, and queryable via **SPARQL** (6 competency + 4 analytical queries).

> **Synthetic Benchmark Note:** Due to current Chinese civil aviation regulations (CAAC), real operational UAM flight data cannot be publicly shared. The benchmark is synthesized based on publicly available regulatory frameworks and operational parameters from published UAM trials. The **[`benchmark_generator/`](./benchmark_generator)** module provides the complete generator with fixed random seed (42), documented parameter distributions, scenario specifications, and a coverage matrix classifying violations as injected vs. emergent.

---

## Repository Structure

```
modules/SkyRwa/
├── README.md                           ← You are here
├── ISWC2026/                           # Paper reproduction artifact
│   ├── README.md                       #   Reviewer-facing guide
│   ├── run.sh / run.ps1                #   One-click reproduction
│   ├── requirements.txt                #   Pinned dependencies
│   ├── reproduce_table5.py             #   Table 5: benchmark dataset (105 flights)
│   ├── reproduce_table6.py             #   Table 6: JSON vs SPARQL baseline
│   ├── reproduce_semantic_baseline.py  #   Table 7: Lifecycle KG vs Flat KG
│   ├── reproduce_table7.py             #   Table 8: governance ablation
│   ├── reproduce_table8.py             #   Table 9: scalability (5–1000 flights)
│   ├── reproduce_table9.py             #   Table 10: SPARQL competency questions
│   ├── reproduce_ontology_quality.py   #   §4.5: OOPS!, consistency, CQ mapping
│   ├── reproduce_case_studies.py       #   §7.8: case studies
│   ├── reproduce_user_study.py         #   §7.9: pilot expert evaluation
│   ├── reproduce_validation.py         #   §5: SHACL validation coverage
│   ├── data/                           #   Pre-generated benchmark data
│   └── outputs/                        #   Experiment result JSONs
│
├── ontology/                           # Domain ontology (Turtle)
│   ├── skyrwa.ttl                      #   13 classes, 26 properties
│   ├── alignments.ttl                  #   PROV-O / DCAT / ODRL / Schema.org
│   └── prefixes.ttl                    #   Shared namespace prefixes
├── shapes/                             # 5 SHACL constraint shapes
├── queries/                            # 10 SPARQL queries (6 CQ + 4 analytical)
│   ├── competency/                     #   CQ1–CQ6
│   └── analytical/                     #   Q1–Q4
├── rdf/                                # RDF serialization layer
│   ├── mapper.py                       #   Domain objects → RDF triples
│   ├── serializer.py                   #   to_turtle(), to_jsonld(), to_graph()
│   └── graph_store.py                  #   In-memory graph store + SPARQL
├── semantic_rules/                     # Governance / promotion / explanation
│   ├── validation_runner.py            #   SHACL validator wrapper
│   ├── governance_rules.py             #   SPARQL-based governance rules
│   ├── promotion_rules.py              #   Product promotion rules
│   └── explanation_rules.py            #   Structured explanation builder
├── productization/                     # Multi-flight aggregation
│   ├── aggregator.py                   #   Group candidates by class
│   ├── product_builder.py              #   Build GovernedProduct
│   └── catalogue.py                    #   Product catalogue + RDF export
├── provenance/                         # Evidence & cryptographic integrity
│   ├── evidence_builder.py             #   SHA-256 digest + evidence chain
│   └── signing.py                      #   Ed25519 signing adapter
├── models/                             # Pydantic data models
├── ingest/                             # Flight data ingestion
├── rights/                             # Governance engine
├── valuation/                          # Rule-based + product-level valuation
├── settlement/                         # Revenue recording & settlement
├── pipeline/                           # End-to-end orchestrator
├── storage/                            # JSON file store
├── benchmark_generator/                 # Reproducible benchmark generator
│   ├── README.md                       #   CAAC data justification + distributions
│   ├── scenario_spec.py                #   10 scenario specs, seed, distributions
│   ├── coverage_matrix.py              #   Violation × scenario coverage matrix
│   └── generate.py                     #   Main generator (seed=42)
├── benchmarks/                         # Legacy benchmark generator (105 flights)
├── experiments/                        # Evaluation scripts
├── tests/                              # 13 pytest test modules
└── examples/                           # Runnable demo
```

---

## Prerequisites

- **Python 3.10+**
- No external API keys required — all experiments run locally
- No GPU required

## Quick Start

```bash
git clone https://github.com/liuyushugreat/SkyNetUamPlatform.git
cd SkyNetUamPlatform/modules/SkyRwa/ISWC2026

pip install -r requirements.txt

# One-click: reproduce all tables and case studies
bash run.sh          # Linux/macOS
# .\run.ps1          # Windows PowerShell
```

See [`ISWC2026/README.md`](./ISWC2026/README.md) for detailed step-by-step instructions.

---

## Paper-to-Code Mapping

| Paper Section | Experiment Script | What it Reproduces |
|---------------|-------------------|--------------------|
| §4.5 — Ontology Quality | `reproduce_ontology_quality.py` | Table 4 + OOPS!-style pitfalls, OWL DL consistency, CQ mapping |
| §5 — Governance & Validation | `reproduce_validation.py` | SHACL + SHACL-SPARQL rule coverage |
| §7.1 — Benchmark Dataset | `reproduce_table5.py` | Table 5: 105 flights, 10 scenarios, 7007 triples |
| §7.2 — JSON Baseline | `reproduce_table6.py` | Table 6: JSON-scan vs SPARQL (4 query tasks) |
| §7.3 — Semantic Baseline | `reproduce_semantic_baseline.py` | Table 7: Lifecycle KG vs Flat KG (4 audit tasks) |
| §7.4 — Governance Ablation | `reproduce_table7.py` | Table 8: Python vs SHACL vs Combined |
| §7.5 — Scalability | `reproduce_table8.py` | Table 9: 5–1000 flights, ~66 triples/flight |
| §7.6 — Robustness | `reproduce_robustness.py` | Multi-run stability, scale sensitivity, threshold sweep |
| §7.7 — SPARQL Queryability | `reproduce_table9.py` | Table 10: CQ1–CQ6 competency questions |
| §7.8 — Case Studies | `reproduce_case_studies.py` | 4 cases: promotion, failure, audit, explainability |
| §7.9 — Expert Evaluation | `reproduce_user_study.py` | Table 11: 4 experts × 4 tasks × 3 interfaces |

---

## Key Results (from Paper)

### Table 8: Governance Ablation (§7.4)

| ID | Violation Type | Python | SHACL | Combined |
|----|----------------|:------:|:-----:|:--------:|
| V1 | Missing evidence digest | – | YES | YES |
| V2 | Missing derivation link | – | YES | YES |
| V3 | Low compliance + tradable | YES | – | YES |
| V4 | High risk + tradable | YES | – | YES |
| V5 | Missing rights on tradable | – | YES* | YES |
| V6 | Incomplete mission + tradable | YES | – | YES |
| | **Detection rate** | **50%** | **50%** | **100%** |

*V5 uses a SHACL-SPARQL constraint (`sh:sparql`), demonstrating the necessity of the SPARQL extension for conditional constraints.

### Table 9: Scalability (§7.5)

| N | Pipeline (ms) | RDF Map (ms) | SHACL (ms) | Triples | T/flight |
|---|:---:|:---:|:---:|:---:|:---:|
| 5 | 0.7 | 2.8 | 78 | 336 | 67 |
| 100 | 6.1 | 48.6 | 397 | 6606 | 66 |
| 1000 | 58.2 | 489.5 | 9634 | 66006 | 66 |

---

## Core Concepts

### Flight Evidence vs Data Asset

| Concept | What it is | Example |
|---------|-----------|---------|
| **Flight Evidence** | Raw attestation record with SHA-256 digest. **Not tradable.** | "UAV-007 flew route R-03, 4980 telemetry points, no violations" |
| **Asset Candidate** | Governed, scored, classified wrapper. Carries `RightsProfile` and `ValuationResultV2`. | Same flight, classified as `route_optimization_sample`, valued at 74.92 USD |
| **Governed Data Product** | Aggregated from multiple candidates. Tradable and licensable. | Weather-operation dataset from 5 flights, valued at 300 USD |
| **Revenue Right** | Downstream entitlement for revenue distribution. | Receipt representing 50% operator share |

### Four-Tier Lifecycle

```
FlightEvidence  ─── governance ──→  AssetCandidate  ─── aggregation ──→  GovernedDataProduct  ─── settlement ──→  RevenueRight
   (Tier 1)                           (Tier 2)                             (Tier 3)                                (Tier 4)
```

---

## Semantic Web / Knowledge Graph

### Ontology

13 OWL classes and 26 properties aligned with four W3C vocabularies:

| SkyRwa Concept | Standard | Aligned Term |
|----------------|----------|-------------|
| FlightEvidence | PROV-O | prov:Entity |
| GovernanceDecision | PROV-O | prov:Activity |
| GovernedDataProduct | DCAT | dcat:Dataset |
| RightsProfile | ODRL 2.2 | odrl:Policy |

### SHACL Validation

Five SHACL shapes enforce structural constraints on FlightEvidence, AssetCandidate, GovernedProduct, SettlementRule, and UsageEvent.

### SPARQL Queries

Six competency questions (CQ1–CQ6) and four analytical queries cover single-entity retrieval, multi-hop traversal, aggregation, and negation-as-failure patterns.

### RDF / JSON-LD / Turtle Export

```python
from SkyRwa.rdf.serializer import to_turtle, to_jsonld, to_graph

ttl = to_turtle(asset_unit)        # Turtle string
jld = to_jsonld(evidence_package)  # JSON-LD string
g   = to_graph(settlement_record)  # rdflib.Graph
```

### Provenance Signing (Ed25519)

```python
from SkyRwa.provenance.signing import Ed25519Signer

signer = Ed25519Signer.generate_keypair("my-signer")
signer.sign_evidence(evidence_package)
assert signer.verify_evidence(evidence_package)
```

---

## Running Tests

```bash
cd SkyNetUamPlatform/modules
python -m pytest SkyRwa/tests/ -v
```

13 test modules cover evidence building, RDF mapping, SHACL validation, SPARQL queries, valuation, revenue splitting, productization, and full pipeline smoke tests.

---

## Citation

```bibtex
@inproceedings{liu2026skyrwa,
  title     = {Modeling Governable Flight-to-Asset Lifecycles
               with Knowledge Graphs, SHACL, and Provenance},
  author    = {Liu, Yushu and Wang, Longbiao and Du, Chenglin and Zhai, Haixiao},
  booktitle = {Proceedings of the 25th International Semantic Web
               Conference (ISWC)},
  year      = {2026}
}
```

## License

Part of [SkyNetUamPlatform](https://github.com/liuyushugreat/SkyNetUamPlatform). Released under MIT License.
