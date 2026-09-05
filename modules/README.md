# Modules Directory

This directory contains modular capabilities for the SkyNetUamPlatform.

## Current Structure

### `SkyKg/`

Modules related to the `research/papers/Knowledge_Engine` paper:

- `SkyNet_Knowledge_Engine/`: ontology, neuro-symbolic reasoning, and LLM explanation
- `voxel_airspace_core/`: voxelized 3D airspace indexing, adaptive octree, and pathfinding

### `SkyRwa/`

Real-World Assetization (RWA) and pricing primitives:

- `valuation.py`: data packet valuation interfaces
- `pricing_engine.py`: dynamic pricing engine
- `economics/pricing.py`: congestion pricing models for airspace voxels

### `SkyFlow/`

Temporal knowledge-graph reasoning for multi-UAV conflict detection. This module
belongs to a different paper track and is intentionally not grouped under `SkyKg/`.


### `Skyrescue/`

Checkable runtime-contract module for typed language-candidate admission,
crash-consistent external-effect commitment, causal local repair, and
SkyRescue-Bench reproduction:

- `skyrescue/benchmark.py`: deterministic scheduler, repair runtime, baselines, explicit reservation release, and runtime/invariant metrics
- `skyrescue/workflow.py`: typed intent compiler, structured failures, recoverable/unrecoverable event profiles, workflow contracts, and local-repair metrics
- `skyrescue/runtime_latency.py` and `scripts/run_runtime_latency_benchmark.py`: matched Native/LangGraph x persistence-off/on configured-stack protocol with identical checkpoint boundaries, five warm-ups, 30 measured repeats, and P50/P95/P99/mean/sample-SD
- `skyrescue/entity_grounding.py`: frozen-ontology, label-isolated place grounding and unresolved-entity execution gate
- `skyrescue/fault_detection.py`: weak-signal online fault detectors, including a post-hoc temporal-causal reservation detector, without label access
- `skyrescue/security.py`: deterministic authorization boundary for security challenge scoring
- `scripts/`: dataset generators, validators, benchmark runners, security scorer, single-seed fault scorer, and multi-seed statistical evaluator
- `scripts/generate_cross_generator_challenge.py` and `scripts/run_cross_generator_challenge.py`: distribution-shift robustness benchmark with frozen detector thresholds
- `scripts/generate_intent_benchmark.py` and `scripts/run_workflow_benchmark.py`: frozen synthetic intent and 172-event workflow-runtime conformance evaluation; expected outcomes remain in an evaluator-only oracle
- `scripts/run_workflow_scale.py`: one connected typed workflow graph at 100/250/500/1,000/2,000/5,000 tasks, with five seeds, five warm-ups, 30 measured passes, and a fresh process per size/seed cell; compilation creates planned state but no `Committed` state or receipt, and the runtime fixture is constructed outside timing
- `skyrescue/langgraph_baseline.py` and `scripts/run_langgraph_baseline.py`: LangGraph StateGraph/SQLite framework embedding around the exact shared application repair and idempotent-receiver contract; the contract semantics are not native LangGraph guarantees
- `skyrescue/durable_runtime.py` and `scripts/run_crash_recovery_experiment.py`: single-host SQLite state machine with `Executing`/`EffectUnknown`, separate invocation/effect/receipt counters, three-valued receiver reconciliation, HMAC-bound receipts, and 90 real child-process terminations across three crash windows
- `skyrescue/core_contract.py`, `skyrescue/devops_adapter.py`, `skyrescue/uav_contract_adapter.py`, and `scripts/run_devops_portability.py`: exact shared contract code exercised by synthetic template-generated UAV and DevOps workloads, with adapter/core identity evidence
- `scripts/reproduce_jss_submission.py`: one-command, network-free JSS evidence reproduction that runs the full test suite and records versions, seeds, commands, and SHA-256 output hashes
- `scripts/run_human_intent_llm_benchmark.py`: fixed-prompt DeepSeek/Qwen evaluation on the 100-case human-authored intent gold set; authoring/annotation independence is partial, and each response is reused for direct JSON, schema, and full-compiler comparisons
- `scripts/run_heldout_llm_blind.py`: label-free instruction-only capture for the frozen 100-case confirmatory set, with resumable raw responses and explicit scenario-card/gold-label exclusion
- `scripts/score_heldout_llm_blind.py` and `scripts/plot_risk_coverage.py`: post-adjudication scoring of frozen responses with seven-field accuracy, 10,000-sample bootstrap intervals, paired significance, risk/coverage diagnostics, and no new API calls
- `scripts/evaluate_entity_grounding.py`: re-evaluates saved model responses with independent target grounding and no new API calls or gold-label access during online inference
- `configs/entity_grounding_freeze_v1.0.0.json` and `scripts/verify_entity_grounding_freeze.py`: lock and verify the ontology, thresholds, and evaluator before held-out confirmatory collection
- `configs/`: synthetic benchmark scale configurations

Immutable JSS submission code and reproduction guide: [`jss-submission-v1.0/modules/Skyrescue`](https://github.com/liuyushugreat/SkyNetUamPlatform/tree/jss-submission-v1.0/modules/Skyrescue). It corresponds to the manuscript's typed admission, crash-consistent external-effect commitment, framework embedding, cross-domain adapter, and commitment-preserving repair evidence.

### `SkyGov/`

LLM-driven multi-agent governance system for UAM regulatory compliance (WISA 2026 / 计算机研究与发展):

- `skygov/agents/`: Four specialized agents (compliance, risk assessment, explanation, audit)
- `skygov/orchestrator/`: DAG-based workflow engine with trust negotiation protocol
- `skygov/rag_pipeline/`: Multi-source retrieval, re-ranking, and context building
- `skygov/governance/`: Decision tracing, hallucination guarding, compliance reports
- `api/`: FastAPI service interface

## Notes

- `SkyKg/` is the paper-focused bundle for the Knowledge Engine work.
- `SkyFlow/` remains separate because it is unrelated to the SkyKG paper.
- `Skyrescue/` is the paper-focused JSS module for SkyRescue. Its compact,
  de-identified frozen input bundle is under `release/jss-submission-v1.0/`;
  large generated telemetry remains outside Git.
- `SkyGov/` extends SkyKG from a single RAG pipeline to a multi-agent LLM governance system.
- `SkyRwa/` remains at the top level because it supports economics and pricing rather than the SkyKG workflow.
