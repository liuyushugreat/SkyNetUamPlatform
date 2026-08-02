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

Symbolically grounded emergency traffic command and SkyRescue-Bench reproduction module:

- `skyrescue/benchmark.py`: deterministic scheduler, repair runtime, and baselines
- `skyrescue/fault_detection.py`: weak-signal online fault detector without label access
- `skyrescue/security.py`: deterministic authorization boundary for security challenge scoring
- `scripts/`: dataset generators, validators, benchmark runners, security scorer, single-seed fault scorer, and multi-seed statistical evaluator
- `scripts/generate_cross_generator_challenge.py` and `scripts/run_cross_generator_challenge.py`: distribution-shift robustness benchmark with frozen detector thresholds
- `configs/`: synthetic benchmark scale configurations

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
- `Skyrescue/` is the paper-focused module for SkyRescue and should hold the reproducible code for the emergency-command paper. Large generated datasets are excluded from Git and should be released through a dataset archive.
- `SkyGov/` extends SkyKG from a single RAG pipeline to a multi-agent LLM governance system.
- `SkyRwa/` remains at the top level because it supports economics and pricing rather than the SkyKG workflow.
