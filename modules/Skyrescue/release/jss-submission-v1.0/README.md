# SkyRescue JSS submission artifact

This directory is the self-contained, offline input bundle for the code tag
`jss-submission-v1.0`. It accompanies the manuscript *SkyRescue: A Checkable
Runtime Contract for Safety-Critical LLM-Generated Workflows*.

## Contents

- `artifact/SkyRescue-Bench/data/intent_synth_v1/`: 300 frozen,
  template-generated intent cases (seed `20260802`).
- `artifact/SkyRescue-Bench/data/security_challenge_v1/`: 600 frozen synthetic
  policy cases (seed `20261001`).
- `artifact/outputs/SkyRescue-Bench-v1.0.0/`: byte-preserved public dataset
  snapshot containing the anonymized HeldOut100 inputs/gold/audit records,
  stored DeepSeek and Qwen responses, model configuration, response hashes,
  and original scored outputs.
- `SHA256SUMS.txt`: hashes for the complete wrapper bundle. The nested frozen
  dataset also retains and verifies its original `SHA256SUMS.txt`.

The nested dataset's historical manifest records the repository URL used when
that dataset was frozen. The submission tag and this wrapper README are the
authoritative code location for the JSS revision; the nested bytes are not
rewritten merely to update a URL.

## Verified environment

- macOS 15.6.1, arm64, Apple M3, 16 GiB RAM
- Python 3.13.2
- Direct dependency versions are in `../../requirements.txt`.
- The fully resolved verification environment is in
  `../../requirements-lock.txt`.

From `modules/Skyrescue`:

```bash
python3 -m venv .venv-skyrescue
source .venv-skyrescue/bin/activate
python -m pip install -r requirements-lock.txt
shasum -a 256 -c release/jss-submission-v1.0/SHA256SUMS.txt
python scripts/reproduce_jss_submission.py \
  --output-dir /tmp/skyrescue-jss-submission-v1.0 \
  --require-clean-tag
```

The reproduction command deliberately does not invoke a model API. It verifies
the frozen grounder and input hashes; runs the full test suite; executes the
compilation/runtime evaluation and LangGraph framework embedding; re-scores stored
HeldOut100 responses with 10,000 bootstrap samples; runs 90 child-process crash
injections; executes matched Native/LangGraph persistence timing, five-seed
single-task-graph scaling, and the UAV/DevOps adapter check; then hashes every
output.

The deterministic event oracle is evaluator-only: native and LangGraph runtime
paths receive observable event fields, not expected outcomes or failure
reasons. LangGraph supplies StateGraph control flow, retry, and checkpointing
around the exact shared application repair and idempotent-receiver contract; the
contract semantics are not native LangGraph guarantees.

## Expected outputs

- `workflow/{compiler_results.csv,runtime_results.csv,workflow_benchmark.json}`
- `langgraph/langgraph_workflow_baseline.json` and checkpoint database
- `heldout100/` bootstrap, risk--coverage, predictions, and summary files
- `crash_recovery/crash_recovery_results.json`
- `runtime_latency/` raw/summary timing CSV and metadata
- `workflow_scale/` raw/run/summary CSV, JSON, and two PDF/PNG figures
- `devops/devops_portability.json` (contains both domain executions)
- `manuscript_tables/table_compilation.csv`
- `manuscript_tables/table_runtime.csv`
- `manuscript_tables/table_latency.csv`
- `manuscript_tables/table_scale.csv`
- `manuscript_tables/table_devops.csv`
- `manuscript_tables/table_heldout_bootstrap.csv`
- `manuscript_tables/risk_coverage.csv`
- `manuscript_figures/` (three embedded-font PDFs)
- `reproduction_manifest.json` with portable commands, input/output SHA-256,
  dependency versions, repository/tag status, and all fixed seeds

The scale cells contain one connected graph with exactly 100, 250, 500, 1,000,
2,000, or 5,000 typed tasks. Each size uses five seeds, five warm-up passes, and
30 measured passes. Compilation builds typed planned state but no `Committed`
state or receipt; the already-executed runtime fixture and synthetic pre-event
receipts are created after compilation and outside event timing.

Fixed seeds are `20260802` (IntentSynth), `20261001` (security challenge),
`20260905` (bootstrap and both adapters), and `20260811`--`20260815` (scale).
Timing distributions and peak RSS are host-sensitive; deterministic counts and
state/effect invariants must match, while timing values should be interpreted as
configured-stack measurements on the stated host.

## Scope

The artifact contains synthetic or research-authored data only. It contains no
API credentials, identifiable expert records, operational flight logs, or live
infrastructure data. Its crash receiver is simulated and key-deduplicating. The
per-workflow operation key is HMAC-derived; the receipt authenticates that key,
issue version, causal parent, and outcome, and the receiver row separately
records the workflow. The evidence-chain hash covers the event kind. The two
domain adapters are template-generated, in-memory workloads. These mechanisms
do not establish distributed exactly-once delivery, operational deployment, or
real-flight validity.
