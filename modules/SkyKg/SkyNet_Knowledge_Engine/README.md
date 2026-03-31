# SkyNet Knowledge Engine

Core implementation for the `research/papers/Knowledge_Engine` paper.

## Included Components

- `ontology/skynet_core.ttl`: RDF/OWL ontology for UAVs, airspace resources, and risk semantics
- `reasoning/neuro_symbolic_reasoner.py`: symbolic reasoning over telemetry and ontology facts
- `llm_agent/deepseek_client.py`: DeepSeek-backed explanation generation
- `llm_agent/risk_explainer.py`: mock explainability demo used for paper-style outputs

## Role in SkyKG

This module provides the neuro-symbolic layer of SkyKG:

- ontology-backed knowledge representation
- SPARQL-driven symbolic retrieval and rule checks
- LLM-assisted natural language explanations

