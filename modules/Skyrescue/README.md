# SkyRescue

SkyRescue is the paper module for intent-driven multi-agent workflow compilation,
runtime repair, and runtime-assured execution in emergency low-altitude command. It
contains a typed workflow compiler, workflow-runtime baselines, a deterministic
domain resource binder, weak-signal fault challenges, and authorization-policy
tests.

## What is included

- `skyrescue/benchmark.py`: SkyRescue runtime and baselines (`greedy`, `cp_sat`, `no_symbol_grounding`, `no_audit`, `full_replan`, `skyrescue`).
- `skyrescue/workflow.py`: typed intent compiler, structured failures, workflow contracts, runtime baselines, and local-repair metrics.
- `skyrescue/entity_grounding.py`: label-isolated contextual place grounding with a frozen emergency-domain ontology and execution gate for unresolved entities.
- `skyrescue/fault_detection.py`: online weak-signal detectors and comparison baselines that do not read fault labels during inference.
- `skyrescue/security.py`: deterministic authorization boundary for security challenge evaluation.
- `scripts/generate_dataset.py`: synthetic emergency-traffic benchmark generator.
- `scripts/generate_security_challenges.py`: authorization challenge generator.
- `scripts/generate_fault_challenge.py`: weak-signal partial-observability fault challenge generator.
- `scripts/generate_cross_generator_challenge.py`: independent distribution-shift generator using heterogeneous baselines, autoregressive noise, gradual faults, intermittent observability, and correlated benign regimes.
- `scripts/run_skyrescue_benchmark.py`: multi-dataset evaluator and 10-seed summary writer.
- `scripts/run_security_challenges.py`: security challenge scorer.
- `scripts/run_fault_challenge.py`: fault challenge scorer.
- `scripts/run_fault_challenge_multiseed.py`: multi-seed generator, scorer, confidence-interval summarizer, and paired significance tester.
- `scripts/run_cross_generator_challenge.py`: frozen-detector cross-generator evaluator with per-seed outputs, confidence intervals, and paired significance tests.
- `scripts/generate_intent_benchmark.py`: frozen 300-case Chinese synthetic-intent generator.
- `scripts/run_workflow_benchmark.py`: compiler and event-driven workflow-runtime evaluator.
- `scripts/run_workflow_scale.py`: 100/500/1,000/2,000 workflow, five-seed state-transition and evidence-hashing scale runner.
- `scripts/run_human_intent_llm_benchmark.py`: frozen DeepSeek/Qwen evaluation on the independently annotated human-instruction gold set.
- `scripts/run_heldout_llm_blind.py`: instruction-only, label-free DeepSeek/Qwen capture runner for the frozen 100-case confirmatory set; rejects inputs containing scenario cards or labels and saves resumable raw-response checkpoints.
- `scripts/score_heldout_llm_blind.py`: post-adjudication scorer for saved held-out responses; validates the blind boundary, reports seven-field accuracy, paired significance, compiler outcomes, and frozen entity-gate diagnostics without making another API request.
- `scripts/evaluate_entity_grounding.py`: offline four-stage re-evaluation of saved LLM responses; gold targets are opened only after independent grounding.
- `configs/entity_grounding_freeze_v1.0.0.json`: immutable hashes, thresholds, and protocol boundary for the held-out confirmatory evaluation.
- `scripts/verify_entity_grounding_freeze.py`: verifies frozen source hashes before any held-out response is scored.

Large generated datasets are intentionally excluded from this Git module. Keep them in the paper workspace or publish them separately through an archival dataset host.

## Quick start

```bash
cd modules/Skyrescue
python -m venv .venv-skyrescue
source .venv-skyrescue/bin/activate
pip install -r requirements.txt
```

Generate a small synthetic benchmark:

```bash
python scripts/generate_dataset.py --config configs/small.json --output /tmp/skyrescue-small
python scripts/validate_dataset.py --dataset /tmp/skyrescue-small
```

Run the main evaluator:

```bash
python scripts/run_skyrescue_benchmark.py \
  --datasets /tmp/skyrescue-small \
  --output-dir /tmp/skyrescue-results/small_run
```

Generate and evaluate the workflow benchmark:

```bash
python scripts/generate_intent_benchmark.py \
  --output /tmp/skyrescue-intent-synth
python scripts/run_workflow_benchmark.py \
  --dataset /tmp/skyrescue-intent-synth \
  --output-dir /tmp/skyrescue-results/workflow
python scripts/run_workflow_scale.py \
  --output-dir /tmp/skyrescue-results/workflow-scale
```

Run the real-LLM human-instruction benchmark with credentials stored outside
the repository. The runner never copies API keys into its outputs and resumes
from per-model JSONL checkpoints:

```bash
python scripts/run_human_intent_llm_benchmark.py \
  --input /path/to/SkyRescue_HumanInstructions_100_GoldStandard_v1.0.0.jsonl \
  --key-file /path/to/key.md \
  --output-dir /tmp/skyrescue-results/human-intent-llm
```

The experiment freezes one prompt, `temperature=0`, `top_p=1`, and one response
per instruction. DeepSeek thinking mode is explicitly disabled so both APIs
obey the same deterministic decoding settings. It requests `deepseek-v4-flash` and
`qwen3-30b-a3b-instruct-2507`. Direct JSON parsing, schema validation, and the
full typed SkyRescue compiler reuse the same raw model response, preventing
sampling differences from confounding the three-stage comparison.

Capture the held-out confirmatory responses before opening A/B annotations or
adjudicated labels. The blind input must contain exactly `instruction_id` and
`instruction_text`; scenario cards and gold fields are rejected:

```bash
PYTHONPATH=. python scripts/run_heldout_llm_blind.py \
  --input /path/to/SkyRescue_EntityGrounding_HeldOut100_BlindInput_v1.0.0.jsonl \
  --key-file /path/to/key.md \
  --output-dir /tmp/skyrescue-results/heldout100-blind
```

This runner freezes an instruction-only prompt, `temperature=0`, `top_p=1`,
`max_tokens=512`, and one response per model and case. It records the direct
JSON, schema, typed-compiler, and frozen entity-grounding-gate outcomes without
reading or scoring any gold label. Final accuracy metrics are computed only
after independent A/B annotation and third-expert adjudication.

After adjudication, score the saved responses without calling either model again:

```bash
PYTHONPATH=. python scripts/score_heldout_llm_blind.py \
  --gold /path/to/SkyRescue_EntityGrounding_HeldOut100_GoldStandard_v1.0.0.jsonl \
  --response-dir /tmp/skyrescue-results/heldout100-blind \
  --output-dir /tmp/skyrescue-results/heldout100-confirmatory
```

The scorer refuses response files that contain scenario-card or gold-label
inputs. It preserves the pre-adjudication raw-response hashes and distinguishes
strict target extraction, entity-gate acceptance/blocking, and exact compiler
outcome accuracy.

Re-evaluate saved responses with the label-isolated entity grounder without
making another API request:

```bash
PYTHONPATH=. python scripts/evaluate_entity_grounding.py \
  --input /path/to/SkyRescue_HumanInstructions_100_GoldStandard_v1.0.0.jsonl \
  --response-dir /path/to/saved-human-intent-llm-responses \
  --output-dir /tmp/skyrescue-results/entity-grounding
```

The grounder accepts only the scenario card, instruction text, and predicted
target. It cannot accept a gold target. The offline evaluator grounds the gold
target separately and compares frozen ontology IDs. An executable candidate
whose place cannot be uniquely grounded is rejected with `UngroundedEntity`.

Before collecting or scoring confirmatory cases, verify that the frozen
grounder and evaluator have not changed:

```bash
python scripts/verify_entity_grounding_freeze.py
```

Run the two challenge scorers:

```bash
python scripts/generate_security_challenges.py --output /tmp/skyrescue-security-challenge
python scripts/run_security_challenges.py --dataset /tmp/skyrescue-security-challenge --output /tmp/skyrescue-results/security_challenge_v1.json

python scripts/generate_fault_challenge.py --output /tmp/skyrescue-fault-challenge
python scripts/run_fault_challenge.py --dataset /tmp/skyrescue-fault-challenge --output /tmp/skyrescue-results/fault_challenge_v1.json

python scripts/run_fault_challenge_multiseed.py \
  --data-dir /tmp/skyrescue-fault-challenge-10seed \
  --output-dir /tmp/skyrescue-results/fault_challenge_v1_10seed

python scripts/run_cross_generator_challenge.py \
  --data-dir /tmp/skyrescue-cross-generator-10seed \
  --output-dir /tmp/skyrescue-results/cross_generator_v1_10seed
```

The fault scorer runs five detectors by default:

| Detector | Purpose |
| --- | --- |
| `single_signal` | high-recall baseline that reports any single abnormal signal |
| `structural_only` | baseline using replay, audit gap, actuator, and reservation signals |
| `persistent_fusion` | conservative temporal baseline requiring sustained weak evidence |
| `skyrescue_fusion` | SkyRescue runtime detector using typed evidence fusion and debounce gating |
| `skyrescue_causal` | post-hoc variant that requires reservation evidence in at least 5 of the latest 7 telemetry samples |

The benchmark evaluator also reports repair P50/P95/P99, scheduler wall time,
peak resident memory, timeout rate, invariant violations, duplicate external
calls, residual reservations, and structured failure reasons. During repair it
explicitly releases the superseded route reservation before committing a
replacement.

## Current paper-scale results

The paper workspace currently contains a 10-seed synthetic evaluation under `SkyRescue-Bench/results/skyrescue_experiments_10seed`.  Summary values should be reported as synthetic benchmark evidence, not real flight evidence.

The frozen `SkyRescue-IntentSynth` v1.0.0 set contains 300 template-generated
Chinese instructions. On that controlled set, the full compiler obtains 1.0000
slot F1, 1.0000 executable-workflow rate on valid cases, and 1.0000 structured-
failure accuracy. The associated 172-workflow event simulation contains 164
recoverable events and eight deterministic unrecoverable boundary cases. The
benchmark reports repair success only over recoverable events and separately
scores whether unrecoverable cases are rejected or escalated with the correct
structured reason. Change ratio and commitment preservation are computed only
over successfully recovered workflows; methods without a typed repair workflow
report these fields as not applicable.

On the frozen 172-workflow sequence, SkyRescue repairs all 164 recoverable
events, correctly rejects or escalates all eight unrecoverable cases, changes
0.2678 of nodes in successfully recovered workflows, preserves all committed
nodes, and emits no duplicate external call. Full replanning reaches the same
recovery and failure-handling rates but changes every node and preserves no
commitment. These are deterministic state-machine conformance results.

These are mechanism-conformance results over generator labels. The benchmark
does not contain a real LLM baseline, instructions collected from emergency
commanders, double-blind human annotation, or inter-annotator agreement.

Key values from the current 10-seed run:

| Method | Completion | On-time rate | Conflict rate | Evidence completeness |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 0.7100 ± 0.1252 | 0.0500 ± 0.0247 | 0.3562 ± 0.2127 | 0.2015 ± 0.0508 |
| CP-SAT assignment + scheduler | 0.9458 ± 0.0087 | 0.5975 ± 0.0890 | 0.0000 ± 0.0000 | 0.9129 ± 0.0135 |
| No symbolic grounding | 0.2913 ± 0.1391 | 0.0050 ± 0.0021 | 0.0000 ± 0.0000 | 0.2897 ± 0.1390 |
| No audit | 1.0000 ± 0.0000 | 0.9340 ± 0.0155 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Full replan | 1.0000 ± 0.0000 | 0.9090 ± 0.0192 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| SkyRescue | 1.0000 ± 0.0000 | 0.9340 ± 0.0155 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 |

Current weak-signal fault challenge (`SkyRescue-FaultChallenge` v1.1.0, 10 seeds, 1,200 synthetic faults, typed event-level overlap; values are mean ± sample standard deviation):

| Detector | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Single signal | 0.1584 ± 0.0068 | 0.9975 ± 0.0040 | 0.2734 ± 0.0100 |
| Structural only | 0.2954 ± 0.0222 | 0.7017 ± 0.0170 | 0.4154 ± 0.0235 |
| Persistent fusion | 0.6640 ± 0.0254 | 0.8325 ± 0.0224 | 0.7386 ± 0.0216 |
| SkyRescue fusion | 0.9863 ± 0.0069 | 0.9858 ± 0.0088 | 0.9861 ± 0.0063 |

The 95% confidence interval for SkyRescue F1 is [0.9815, 0.9906]. Its F1 advantage over each baseline is significant under a two-sided exact paired sign-flip test with Holm correction (`p = 0.005859`). The single-signal detector has slightly higher recall but very low precision, exposing the expected alarm-volume tradeoff. Per-seed JSON, per-fault-type CSV, and significance tables are written by `scripts/run_fault_challenge_multiseed.py`.

Frozen-threshold cross-generator challenge (`SkyRescue-CrossGenerator` v1.0.0, 10 new seeds and 1,200 faults):

| Detector | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Single signal | 0.3906 ± 0.0245 | 1.0000 ± 0.0000 | 0.5614 ± 0.0253 |
| Structural only | 0.3782 ± 0.0152 | 0.6667 ± 0.0000 | 0.4824 ± 0.0122 |
| Persistent fusion | 0.6486 ± 0.0171 | 0.9850 ± 0.0117 | 0.7820 ± 0.0129 |
| SkyRescue fusion | 0.7148 ± 0.0157 | 0.9975 ± 0.0040 | 0.8327 ± 0.0114 |

SkyRescue fusion remains significantly above all three baselines (`Holm p = 0.005859` for F1), but its F1 drops by 0.1534 from the internal challenge. Reservation-conflict F1 is 0.4065 ± 0.0298, identifying sensitivity to persistent benign reservation-score regimes. This negative transfer result is reported without retuning the detector.

## Reproducibility notes

- All benchmark data are synthetic and generated from fixed seeds.
- `direct_text` is a model-free non-workflow control, not a Direct-LLM result.
- IntentSynth labels come from the generator and are not a human gold set.
- Fault labels are withheld from online scheduling and detection, then opened only for offline scoring.
- Cross-generator evaluation freezes all detector thresholds learned from the original challenge and changes the generator family rather than only the random seed.
- The CP-SAT baseline is a centralized resource-assignment baseline followed by the shared reservation scheduler; it is not a proof of global optimality for the full traffic-control problem.
- The security challenge evaluates deterministic policy coverage rather than operational flight safety.
- `skyrescue_causal` was added after inspecting the v1.0.0 reservation-conflict
  failure mode. It must be reported as a post-hoc error-driven improvement, not
  as part of the original frozen-detector comparison.
- The first frozen place ontology resolves 41 of 100 gold targets. On the saved
  100-response runs, anchor accuracy is 0.52 for DeepSeek and 0.45 for Qwen,
  compared with strict normalized-text accuracy of 0.20 and 0.18. This is a
  conservative reproducible baseline: unresolved targets are blocked, and the
  ontology coverage gap must be reported rather than treated as model error.
- Because this ontology was implemented after prior inspection of benchmark
  errors, the 200-response re-evaluation is post-hoc development evidence. A
  confirmatory paper claim requires freezing the ontology and thresholds before
  evaluation on an unseen human-instruction set.
