from pathlib import Path

from skyrescue.durable_runtime import DurableWorkflowRuntime


def test_sink_reconciliation_after_pre_receipt_effect(tmp_path: Path):
    database = tmp_path / "runtime.sqlite"
    runtime = DurableWorkflowRuntime(database)
    runtime.start("wf-1")
    operation = runtime._ensure_operation("wf-1")
    runtime._external_effect(operation["idempotency_key"])
    runtime.close()

    resumed = DurableWorkflowRuntime(database)
    resumed.execute("wf-1")
    state = resumed.inspect("wf-1")
    resumed.close()

    assert state["workflow_status"] == "Committed"
    assert state["effect_count"] == 1
    assert state["workflow_version"] == 2
    assert state["reservation_consistent"]
    assert state["evidence_chain_continuous"]
