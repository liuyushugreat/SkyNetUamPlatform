import json
from pathlib import Path
import subprocess
import sys

import pytest

from skyrescue.durable_runtime import (
    CRASH_EXIT_CODE,
    CrashPoint,
    DurableWorkflowRuntime,
    OperationState,
)


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_crash_recovery_experiment.py"


def run_crash(database: Path, workflow_id: str, crash_point: CrashPoint) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--worker",
            "--database",
            str(database),
            "--workflow-id",
            workflow_id,
            "--crash-point",
            crash_point.value,
        ],
        check=False,
    )
    assert completed.returncode == CRASH_EXIT_CODE


def resume_and_inspect(database: Path, workflow_id: str) -> dict:
    runtime = DurableWorkflowRuntime(database)
    try:
        runtime.start(workflow_id)
        runtime.execute(workflow_id)
        return runtime.inspect(workflow_id)
    finally:
        runtime.close()


def assert_single_committed_effect(state: dict) -> None:
    assert state["workflow_status"] == OperationState.COMMITTED.value
    assert state["operation_state"] == OperationState.COMMITTED.value
    assert state["invoke_count"] == 1
    assert state["effect_count"] == 1
    assert state["receipt_count"] == 1
    assert state["workflow_version"] == 2
    assert state["issue_version"] == 1
    assert state["causal_parent"]
    assert state["receiver_receipt_valid"]
    assert state["local_receipt_valid"]
    assert state["reservation_consistent"]
    assert state["evidence_chain_continuous"]


def test_crash_before_external_call(tmp_path: Path):
    database = tmp_path / "w1.sqlite"
    run_crash(database, "wf-w1", CrashPoint.BEFORE_EXTERNAL_CALL)

    pre_recovery = DurableWorkflowRuntime(database)
    try:
        state = pre_recovery.inspect("wf-w1")
        assert state["operation_state"] == OperationState.EXECUTING.value
        assert state["invoke_count"] == 0
        assert state["effect_count"] == 0
        assert state["receipt_count"] == 0
        assert "issue_prepared" in state["evidence_kinds"]
        assert "external_call_issued" not in state["evidence_kinds"]
        assert "receiver_effect_applied" not in state["evidence_kinds"]
    finally:
        pre_recovery.close()

    assert_single_committed_effect(resume_and_inspect(database, "wf-w1"))


def test_crash_after_effect_before_receipt(tmp_path: Path):
    database = tmp_path / "w2.sqlite"
    run_crash(database, "wf-w2", CrashPoint.AFTER_EFFECT_BEFORE_RECEIPT)

    pre_recovery = DurableWorkflowRuntime(database)
    try:
        state = pre_recovery.inspect("wf-w2")
        assert state["operation_state"] == OperationState.EXECUTING.value
        assert state["invoke_count"] == 1
        assert state["effect_count"] == 1
        assert state["receipt_count"] == 0
        assert "issue_prepared" in state["evidence_kinds"]
        assert "external_call_issued" not in state["evidence_kinds"]
        assert "receiver_effect_applied" in state["evidence_kinds"]
    finally:
        pre_recovery.close()

    assert_single_committed_effect(resume_and_inspect(database, "wf-w2"))


def test_crash_after_receipt_persisted(tmp_path: Path):
    database = tmp_path / "w3.sqlite"
    run_crash(database, "wf-w3", CrashPoint.AFTER_RECEIPT_PERSISTED)
    assert_single_committed_effect(resume_and_inspect(database, "wf-w3"))


def test_reconcile_effect_already_occurred(tmp_path: Path):
    runtime = DurableWorkflowRuntime(tmp_path / "occurred.sqlite")
    try:
        runtime.start("wf-occurred")
        runtime._ensure_operation("wf-occurred")
        runtime._precheck("wf-occurred")
        operation = runtime._mark_executing("wf-occurred")
        runtime._invoke_external_effect(operation["idempotency_key"])

        assert runtime.reconcile("wf-occurred") == OperationState.COMMITTED.value
        assert_single_committed_effect(runtime.inspect("wf-occurred"))
    finally:
        runtime.close()


def test_reconcile_effect_not_occurred(tmp_path: Path):
    runtime = DurableWorkflowRuntime(tmp_path / "absent.sqlite")
    try:
        runtime.start("wf-absent")
        runtime._ensure_operation("wf-absent")
        runtime._precheck("wf-absent")
        runtime._mark_executing("wf-absent")

        assert runtime.reconcile("wf-absent") == OperationState.PRECHECKED.value
        before_retry = runtime.inspect("wf-absent")
        assert before_retry["effect_count"] == 0
        assert before_retry["receipt_count"] == 0

        runtime.execute("wf-absent")
        assert_single_committed_effect(runtime.inspect("wf-absent"))
    finally:
        runtime.close()


def test_no_duplicate_effect_for_non_idempotent_action(tmp_path: Path):
    database = tmp_path / "non-idempotent.sqlite"
    run_crash(database, "wf-once", CrashPoint.AFTER_EFFECT_BEFORE_RECEIPT)

    first = resume_and_inspect(database, "wf-once")
    second = resume_and_inspect(database, "wf-once")
    assert_single_committed_effect(first)
    assert_single_committed_effect(second)


def test_reconcile_unavailable_escalates_without_retry(tmp_path: Path):
    runtime = DurableWorkflowRuntime(tmp_path / "unknown.sqlite")
    try:
        runtime.start("wf-unknown")
        runtime._ensure_operation("wf-unknown")
        runtime._precheck("wf-unknown")
        runtime._mark_executing("wf-unknown")
        assert runtime.reconcile("wf-unknown", receiver_query_available=False) == OperationState.HUMAN_ESCALATED.value
        state = runtime.inspect("wf-unknown")
        assert state["invoke_count"] == 0
        assert state["effect_count"] == 0
        assert state["receipt_count"] == 0
    finally:
        runtime.close()


def prepare_receiver_effect(
    runtime: DurableWorkflowRuntime, workflow_id: str
) -> tuple[object, str]:
    runtime.start(workflow_id)
    runtime._ensure_operation(workflow_id)
    runtime._precheck(workflow_id)
    operation = runtime._mark_executing(workflow_id)
    receipt = runtime._invoke_external_effect(operation["idempotency_key"])
    return operation, receipt


def test_receiver_receipt_binds_key_issue_version_and_causal_parent(tmp_path: Path):
    runtime = DurableWorkflowRuntime(tmp_path / "bound-receipt.sqlite")
    try:
        operation, receipt = prepare_receiver_effect(runtime, "wf-bound")
        sink = runtime.connection.execute(
            "SELECT * FROM external_sink WHERE idempotency_key = ?",
            (operation["idempotency_key"],),
        ).fetchone()
        claims = json.loads(receipt)

        assert sink["workflow_version"] == operation["issue_version"] == 1
        assert sink["causal_parent"] == operation["causal_parent"]
        assert claims["idempotency_key"] == operation["idempotency_key"]
        assert claims["workflow_issue_version"] == operation["issue_version"]
        assert claims["causal_parent"] == operation["causal_parent"]
        assert runtime._receipt_is_valid(
            receipt,
            operation["idempotency_key"],
            operation["issue_version"],
            operation["causal_parent"],
        )
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("column", "tampered_value"),
    [
        ("receipt", "forged-receipt"),
        ("workflow_version", 99),
        ("causal_parent", "forged-parent"),
    ],
)
def test_tampered_receiver_receipt_fails_safe(
    tmp_path: Path, column: str, tampered_value: str | int
):
    runtime = DurableWorkflowRuntime(tmp_path / f"tampered-{column}.sqlite")
    try:
        operation, _ = prepare_receiver_effect(runtime, f"wf-tampered-{column}")
        assert column in {"receipt", "workflow_version", "causal_parent"}
        runtime.connection.execute(
            f"UPDATE external_sink SET {column} = ? WHERE idempotency_key = ?",
            (tampered_value, operation["idempotency_key"]),
        )
        runtime.connection.commit()

        assert runtime.reconcile(operation["workflow_id"]) == OperationState.HUMAN_ESCALATED.value
        state = runtime.inspect(operation["workflow_id"])
        assert state["operation_state"] == OperationState.HUMAN_ESCALATED.value
        assert state["workflow_status"] == OperationState.HUMAN_ESCALATED.value
        assert state["workflow_version"] == 1
        assert state["receipt_count"] == 0
        assert not state["receiver_receipt_valid"]
        assert not state["local_receipt_valid"]
        assert "receipt_verification_failed" in state["evidence_kinds"]
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("column", "tampered_value"),
    [
        ("idempotency_key", "forged-key"),
        ("issue_version", 99),
        ("causal_parent", "forged-parent"),
    ],
)
def test_tampered_durable_issue_identity_fails_safe(
    tmp_path: Path, column: str, tampered_value: str | int
):
    runtime = DurableWorkflowRuntime(tmp_path / f"tampered-issue-{column}.sqlite")
    try:
        operation, _ = prepare_receiver_effect(runtime, f"wf-issue-{column}")
        assert column in {"idempotency_key", "issue_version", "causal_parent"}
        runtime.connection.execute(
            f"UPDATE operations SET {column} = ? WHERE workflow_id = ?",
            (tampered_value, operation["workflow_id"]),
        )
        runtime.connection.commit()

        assert runtime.reconcile(operation["workflow_id"]) == OperationState.HUMAN_ESCALATED.value
        state = runtime.inspect(operation["workflow_id"])
        assert state["operation_state"] == OperationState.HUMAN_ESCALATED.value
        assert state["workflow_status"] == OperationState.HUMAN_ESCALATED.value
        assert state["workflow_version"] == 1
        assert state["receipt_count"] == 0
        assert not state["local_receipt_valid"]
        assert "issue_verification_failed" in state["evidence_kinds"]
    finally:
        runtime.close()


def test_forged_direct_receipt_cannot_commit(tmp_path: Path):
    runtime = DurableWorkflowRuntime(tmp_path / "forged-direct.sqlite")
    try:
        operation, _ = prepare_receiver_effect(runtime, "wf-forged-direct")

        assert not runtime._persist_receipt(
            operation["workflow_id"], "forged-receipt", "receipt_persisted"
        )
        state = runtime.inspect(operation["workflow_id"])
        assert state["operation_state"] == OperationState.HUMAN_ESCALATED.value
        assert state["workflow_status"] == OperationState.HUMAN_ESCALATED.value
        assert state["workflow_version"] == 1
        assert state["receipt_count"] == 0
        assert not state["local_receipt_valid"]
    finally:
        runtime.close()


def test_tampered_evidence_kind_breaks_chain_validation(tmp_path: Path):
    runtime = DurableWorkflowRuntime(tmp_path / "tampered-evidence-kind.sqlite")
    try:
        runtime.start("wf-kind-tamper")
        runtime.execute("wf-kind-tamper")
        assert runtime.inspect("wf-kind-tamper")["evidence_chain_continuous"]

        runtime.connection.execute(
            """UPDATE evidence SET kind = 'forged_event_kind'
               WHERE workflow_id = ? AND kind = 'receiver_effect_applied'""",
            ("wf-kind-tamper",),
        )
        runtime.connection.commit()

        state = runtime.inspect("wf-kind-tamper")
        assert not state["evidence_chain_continuous"]
        assert "forged_event_kind" in state["evidence_kinds"]
    finally:
        runtime.close()
