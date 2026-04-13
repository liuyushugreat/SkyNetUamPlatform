# ISWC 2026 Development Notes

## Paper Title

**From Flight Evidence to Governable Data Assets: A Knowledge Graph–Driven
Flight-to-Asset Pipeline for Urban Air Mobility**

---

## Implementation → Paper Section Mapping

| Implementation | Paper Section |
|---|---|
| Ontology (`ontology/skyrwa.ttl`, `alignments.ttl`) | §3 Ontology and KG Model |
| PROV-O / DCAT / ODRL alignments | §3.2 Vocabulary Alignment |
| RDF mapper / serializer (`rdf/`) | §3.3 Knowledge Graph Construction |
| SHACL shapes (`shapes/`) | §4 Governance and Validation |
| Governance rules (`semantic_rules/governance_rules.py`) | §4.1 Rule-based Governance |
| Promotion rules (`semantic_rules/promotion_rules.py`) | §4.2 Promotion Lifecycle |
| Explanation builder (`semantic_rules/explanation_rules.py`) | §5.2 Value Explanation |
| Productization (`productization/`) | §5 Productization and Value Explanation |
| Product valuation (`valuation/product_valuation.py`) | §5.1 Multi-level Valuation |
| Ed25519 signing (`provenance/signing.py`) | §3.4 Provenance and Integrity |
| Benchmark generator (`benchmarks/generate_benchmark.py`) | §6.1 Dataset |
| SPARQL competency queries (`queries/`) | §6.2 Queryability Evaluation |
| `eval_validation.py` | §6.3 Validation Coverage |
| `eval_queryability.py` | §6.4 JSON vs SPARQL Comparison |
| `eval_overhead.py` | §6.5 Performance Overhead |
| `eval_case_studies.py` | §6.6 Case Studies |

---

## Key Design Decisions

### 1. Four-tier asset lifecycle

```
FlightEvidence → AssetCandidate → GovernedDataProduct → RevenueRight
```

This layering is the central contribution: raw evidence is NOT directly tradable.

### 2. Dual rule representation

Python rules (fast, operational) + SPARQL/SHACL rules (auditable, publishable).
The paper argues this duality provides both runtime efficiency and academic rigor.

### 3. Explanation as first-class citizen

Every governance decision and valuation score has a structured explanation
object that can be serialized to RDF and queried via SPARQL.

### 4. Minimal but real cryptography

Ed25519 signatures replace placeholder strings, enabling a verifiable
provenance chain without heavyweight PKI infrastructure.

---

## What's Been Completed

### P0 (Must-have)
- [x] Domain ontology with 12 classes, 20+ properties
- [x] Vocabulary alignment (PROV-O, DCAT, ODRL, Schema.org)
- [x] RDF mapper for all core domain objects
- [x] Turtle and JSON-LD serialization
- [x] 5 SHACL shape files with constraint validation
- [x] 6 competency questions + 4 analytical queries
- [x] SHACL validation runner (Python integration)
- [x] Benchmark generator (30 flights, 8 scenarios)
- [x] 4 evaluation scripts

### P1 (Strongly recommended)
- [x] Multi-flight productization (aggregator, builder, catalogue)
- [x] Product-level valuation with explanation
- [x] Ed25519 provenance signing
- [x] Semantic governance / promotion / explanation rules

---

## Remaining Work (P2 / Future)

- [ ] Triple store adapter (e.g., Fuseki, GraphDB)
- [ ] Interactive graph visualization / catalogue UI
- [ ] External data marketplace adapter
- [ ] Advanced ODRL policy reasoning
- [ ] Cross-pipeline lineage with SkyRoute / SkyShield modules
- [ ] Formal ontology evaluation (OntoClean, OOPS!)
- [ ] Larger benchmark (100+ flights, multi-region)
- [ ] User study on explanation quality

---

## How to Reproduce Experiments

```bash
cd SkyNetUamPlatform/modules

# Generate benchmark data
python -m SkyRwa.benchmarks.generate_benchmark

# Run all experiments
python -m SkyRwa.experiments.eval_validation
python -m SkyRwa.experiments.eval_queryability
python -m SkyRwa.experiments.eval_overhead
python -m SkyRwa.experiments.eval_case_studies

# Run SPARQL queries
python -m SkyRwa.experiments.run_queries

# Run tests
python -m pytest SkyRwa/tests/ -v
```
