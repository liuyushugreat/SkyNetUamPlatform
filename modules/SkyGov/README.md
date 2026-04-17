# SkyGov: Evidence-Driven Multi-Agent Collaborative Reasoning for UAM Compliance Governance

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Target: 计算机研究与发展 / WISA 2026](https://img.shields.io/badge/Target-CRAD%20%2F%20WISA%202026-orange.svg)]()

> **SkyGov** — 面向低空交通合规治理的证据驱动多智能体协同推理系统

---

## Overview

**SkyGov** is a multi-agent LLM-driven governance system for regulatory compliance in Urban Air Mobility (UAM). It upgrades the single-pipeline RAG approach of [SkyKG](../SkyKg/) into a four-agent collaborative workflow with evidence-grounded reasoning, hard-rule veto, explanation auditing, and decision traceability.

### Four Specialized Agents

| Agent | Role | Key Mechanism |
|-------|------|---------------|
| **ComplianceAgent** | Hard-rule checking via SPARQL | Deterministic rule matching with veto power |
| **RiskAssessmentAgent** | Semantic risk reasoning via RAG | KG-retrieval + LLM inference (extends SkyKG) |
| **ExplanationAgent** | Traceable explanation generation | Rule-grounded NL generation with citation links |
| **AuditAgent** | Real-time quality control | RAR/LEC/UCR scoring, hallucination interception |

### Design Principles

1. **Retrieve-then-Reason** — All LLM inputs are grounded by KG retrieval (no free-form generation)
2. **Veto-capable Compliance** — The ComplianceAgent can override any other agent's output
3. **Auditable Decisions** — Every reasoning step is traced back to specific regulation clause IDs
4. **Quality-gated Trust** — The AuditAgent scores every output; low-quality results trigger re-retrieval or human escalation

### Governance Loop

```text
Scenario Input
   ↓
ComplianceAgent (SPARQL hard rules, veto)
   ↓ pass
RiskAssessmentAgent (KG-RAG semantic reasoning)
   ↓
ExplanationAgent (rule-grounded explanation)
   ↓
AuditAgent (RAR / LEC / UCR scoring)
   ↓ pass / retry
TrustProtocol (veto + gate + weighted vote)
   ↓
Final Decision + Trace + Compliance Report
```

---

## Repository Structure

```
modules/SkyGov/
├── README.md                          ← You are here
├── skygov/                            # Core Python package
│   ├── agents/
│   │   ├── base_agent.py              # Abstract agent interface
│   │   ├── compliance_agent.py        # SPARQL hard-rule checking (veto power)
│   │   ├── risk_assessment_agent.py   # KG+RAG semantic risk reasoning
│   │   ├── explanation_agent.py       # Rule-grounded explanation generation
│   │   └── audit_agent.py            # RAR/LEC/UCR quality scoring
│   ├── orchestrator/
│   │   ├── workflow_engine.py         # DAG-based agent workflow engine
│   │   ├── task_graph.py              # Pre-defined governance task graphs
│   │   └── trust_protocol.py          # Hierarchical trust protocol (veto/gate/vote)
│   ├── rag_pipeline/
│   │   ├── multi_source_retriever.py  # Multi-source KG retrieval
│   │   ├── chunk_ranker.py            # Retrieved chunk re-ranking
│   │   └── context_builder.py         # LLM context window management
│   ├── governance/
│   │   ├── decision_tracer.py         # Decision chain traceability
│   │   ├── hallucination_guard.py     # KG-based fact checking for LLM outputs
│   │   └── compliance_report.py       # Auditable compliance report generator
│   ├── utils/
│   │   └── metrics.py                 # RAR, LEC, UCR metric computation
│   └── config.py                      # Centralized configuration
├── api/
│   ├── fastapi_app.py                 # FastAPI service
│   └── schemas.py                     # Request/response models
├── configs/
│   └── default.yaml                   # Default configuration
├── scripts/
│   ├── run_governance.py              # Single-scenario governance demo
│   ├── run_benchmark.py               # Batch benchmark evaluation
│   ├── run_ablation.py                # Prompt / module ablations and baselines
│   └── run_full_eval.py               # End-to-end metrics, robustness, sensitivity
├── outputs/                           # JSON / CSV evaluation outputs
├── tests/
│   ├── test_agents.py
│   ├── test_workflow.py
│   └── test_governance.py
└── requirements.txt
```

---

## Quick Start

```bash
cd SkyNetUamPlatform/modules/SkyGov

# Install dependencies
pip install -r requirements.txt

# Set DeepSeek API key (Linux/macOS)
export DEEPSEEK_API_KEY="your_key_here"

# PowerShell
$env:DEEPSEEK_API_KEY="your_key_here"

# Run single governance scenario
python scripts/run_governance.py

# Run benchmark evaluation
python scripts/run_benchmark.py --scenarios 100

# Run ablation / baseline study
python scripts/run_ablation.py --scenarios 100 --mock

# Run full evaluation suite
python scripts/run_full_eval.py --scenarios 1000
```

### Key Configuration

Default settings are defined in [`configs/default.yaml`](./configs/default.yaml):

| Parameter | Default |
|-----------|---------|
| LLM model | `deepseek-chat` |
| Retrieval `top_k` | `5` |
| Audit `rar_threshold` | `0.8` |
| Audit `ucr_threshold` | `0.1` |
| Workflow `max_retries` | `2` |
| Trust weights | `risk=0.6`, `explanation=0.4` |

### Evaluation Outputs

The latest evaluation pipeline writes structured outputs to `outputs/`, including:

- `benchmark_summary.json`
- `ablation_summary.json`
- `ablation_details.csv`
- `e2e_metrics.json`
- `param_sensitivity.json`
- `robustness.json`
- `error_analysis.json`

---

## Relationship to Doctoral Research

| Dimension | Description |
|-----------|-------------|
| **Extends SkyKG** | Single RAG pipeline → Multi-agent workflow with quality control |
| **Reuses** | SkyKG ontology (`skynet_core.ttl`), SPARQL reasoner, DeepSeek LLM client |
| **Complements SkyGrid** | SkyGov handles LLM-level governance logic; SkyGrid handles compute-level deployment |
| **Target Venue** | 《计算机研究与发展》 WISA 2026 专题 — 大模型驱动的智能信息系统 |

### Latest Experimental Scope

| Evaluation | Current coverage |
|------------|------------------|
| Hard-rule compliance | 8 SPARQL rules |
| Scenario scale | 1000 synthetic UAM scenarios |
| Real API check | 100 DeepSeek API calls |
| End-to-end metrics | Accuracy / Precision / Recall / F1 / confusion matrix |
| Robustness | rule deletion, prompt degradation, no-audit condition |
| Sensitivity | `theta_RAR`, `M`, trust weights |

---

## Citation

```bibtex
@article{liu2026skygov,
  title   = {SkyGov：面向低空交通合规治理的证据驱动多智能体协同推理系统},
  author  = {刘玉书 and 王龙标 and 杜承林 and 翟海潇},
  journal = {计算机研究与发展},
  year    = {2026},
  note    = {under review}
}
```

## License

This project is licensed under the Apache License 2.0.
