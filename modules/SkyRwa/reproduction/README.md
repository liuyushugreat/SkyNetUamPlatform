# SkyRwa Reproduction Artifact

**Paper:** *Modeling Governable Flight-to-Asset Lifecycles with Knowledge Graphs, SHACL, and Provenance* (submitted to the **Journal of Web Semantics**)

This directory is the **self-contained reproduction pipeline** for the paper. It lives inside the larger [SkyNetUamPlatform](https://github.com/liuyushugreat/SkyNetUamPlatform) repository but runs independently.

---

## Quick Start

```bash
git clone https://github.com/liuyushugreat/SkyNetUamPlatform.git
cd SkyNetUamPlatform/modules/SkyRwa/reproduction

pip install -r requirements.txt

bash run.sh          # Linux/macOS
# .\run.ps1          # Windows PowerShell
```

- **Expected total runtime: ~10 minutes** (step 5 alone takes ~6 minutes)
- **Python 3.10+**, no GPU, no API keys; exact pinned versions in `../requirements.txt`
- Results: console tables + JSON in `outputs/`, benchmark data in `data/`
- Clean-room alternative: `docker build -t skyrwa-artifact ..` then `docker run --rm skyrwa-artifact` (from `modules/SkyRwa/`)

---

## Script → Paper Mapping (13 pipeline steps)

| # | Script | Paper item | Expected result |
|--:|--------|------------|-----------------|
| 1 | `reproduce_table5.py` | Table 5 (§7.1) | 105 flights, 10 scenarios, 7 007 triples, 45 tradable |
| 2 | `reproduce_table6.py` | Table 6 (§7.2) | JSON-scan vs SPARQL on 4 audit tasks |
| 3 | `reproduce_semantic_baseline.py` | Table 7 (§7.3) | lifecycle KG complete on A1–A4; flat KG partial or unanswerable |
| 4 | `reproduce_table7.py` | Tables 8 & 9 (§7.4) | Python 67%, SHACL 50%, Combined 100% |
| 5 | `reproduce_scoring_context.py` | Tables 8 (+ctx column) & 10 (§7.4) | +8 triples/flight, ~4.2× pySHACL cost; extended contract alone reaches 100% |
| 6 | `reproduce_table8.py` | Table 11 (§7.5) | ~66 triples/flight; pySHACL dominates at 1 000 flights |
| 7 | `reproduce_shacl_engines.py` | Table 11 rudof columns + Fig. 2 (§7.5) | rudof ~16× faster end-to-end at 1 000 flights (SHACL Core subset only); both engines conform |
| 8 | `reproduce_robustness.py` | §7.6 | identical counts across 5 seeds; threshold sweep 7.6–22.9% |
| 9 | `reproduce_competency.py` | Table 12 (§7.7) | **12/12 CQs correct** on the enriched 9 035-triple audit graph |
| 10 | `reproduce_walkthrough.py` | §7.8 | regenerates `outputs/walkthrough_generated.tex` (the paper `\input`s it) |
| 11 | `reproduce_validation.py` | §5 | SHACL + governance rule coverage |
| 12 | `reproduce_ontology_quality.py` | §4.5 + Appendix A (Table 13) | pitfall scan, OWL DL consistency, 12-CQ construct mapping |
| 13 | `port_autonomous_vehicle/run_av_port.py` | §8.1 | AV port validates conforming; `av_port_result.json` |

Not in the pipeline (need network access): `../evaluation/run_oops_evaluation.py` and `../evaluation/run_foops_evaluation.py` regenerate the OOPS!/FOOPS! reports quoted in §4.5. Supplementary (not cited by the paper): `reproduce_case_studies.py`, `reproduce_user_study.py`, `reproduce_table9.py` (superseded by `reproduce_competency.py`).

---

## Key Expected Results

### Table 8 — Validation-layer coverage analysis (§7.4)

| ID | Violation type | Python | SHACL | SHACL+ctx | Combined |
|----|----------------|:------:|:-----:|:---------:|:--------:|
| V1 | Missing evidence digest | YES† | YES | YES | YES |
| V2 | Missing derivation link | – | YES | YES | YES |
| V3 | Low compliance + tradable | YES | – | YES | YES |
| V4 | High risk + tradable | YES | – | YES | YES |
| V5 | Missing rights on tradable | – | YES* | YES* | YES |
| V6 | Incomplete mission + tradable | YES | – | YES | YES |
| | **Detection rate** | **67%** | **50%** | **100%** | **100%** |

\* via a `sh:sparql` conditional constraint.  
† via rule GOV-003, deliberately redundant with `FlightEvidenceShape`.

### Table 12 — Competency-question verification (§7.7)

All 12 CQs return exactly the expected result sets (ground truth computed
independently from the Python domain objects): 45 tradable (CQ1), 0 orphan
evidence (CQ7, negative), 60 never-productized candidates (CQ8, negative),
5 post-settlement usage events (CQ9, temporal), 15 flights in the incident
window (CQ10, temporal), and 2 products in each cross-tier aggregation
(CQ11–12).

### Table 11 — Scalability (§7.5)

| N | Pipeline (ms) | RDF map (ms) | pySHACL (ms) | rudof e2e (ms) | Triples |
|---|:---:|:---:|:---:|:---:|:---:|
| 5 | ~0.7 | ~3 | ~80 | ~90 | ~340 |
| 100 | ~6 | ~50 | ~400 | ~120 | ~6 600 |
| 1000 | ~60 | ~490 | ~3 900 | ~240 | ~66 000 |

Timings vary with hardware; ratios and triple counts are stable.

---

## Tests

```bash
cd SkyNetUamPlatform/modules
python -m pytest SkyRwa/tests -q      # 101 tests
```

CI runs the fast subset (≤ 105 flights) on every push: `.github/workflows/skyrwa-ci.yml` at the repository root.
