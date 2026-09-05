"""SQLite-backed crash recovery for one externally visible workflow action.

The receiver is idempotent by operation key.  An invocation and its effect are
counted separately so a retry can be observed without being mistaken for a
duplicate real-world effect.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
from enum import Enum
from pathlib import Path
from typing import Any


CRASH_EXIT_CODE = 70

# These domain-separated keys model credentials held by the issue authority and
# receiver.  The prototype deliberately uses only the Python standard library;
# a deployment would keep the corresponding secrets outside the workflow store
# (or use receiver signatures with a pinned public key).
_ISSUE_AUTHENTICATION_KEY = b"skyrescue-jss-prototype-issue-key-v1"
_RECEIVER_AUTHENTICATION_KEY = b"skyrescue-jss-prototype-receiver-key-v1"


class OperationState(str, Enum):
    PROPOSED = "Proposed"
    PRECHECKED = "Prechecked"
    EXECUTING = "Executing"
    EFFECT_UNKNOWN = "EffectUnknown"
    COMMITTED = "Committed"
    HUMAN_ESCALATED = "HumanEscalated"


class CrashPoint(str, Enum):
    BEFORE_EXTERNAL_CALL = "before_external_call"
    AFTER_EFFECT_BEFORE_RECEIPT = "after_effect_before_receipt"
    AFTER_RECEIPT_PERSISTED = "after_receipt_persisted"


def _hash(previous: str, workflow_id: str, kind: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        {"kind": kind, "payload": payload, "workflow_id": workflow_id},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(f"{previous}|{encoded}".encode()).hexdigest()


class DurableWorkflowRuntime:
    """Persist issue/receipt state and reconcile a simulated external receiver."""

    def __init__(self, database: Path):
        self.database = database
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(database)
        self.connection.row_factory = sqlite3.Row
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS workflows (
              workflow_id TEXT PRIMARY KEY, version INTEGER NOT NULL, status TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reservations (
              workflow_id TEXT NOT NULL, slot TEXT NOT NULL, state TEXT NOT NULL,
              PRIMARY KEY (workflow_id, slot)
            );
            CREATE TABLE IF NOT EXISTS external_sink (
              idempotency_key TEXT PRIMARY KEY, workflow_id TEXT NOT NULL,
              workflow_version INTEGER NOT NULL, causal_parent TEXT NOT NULL,
              effect_count INTEGER NOT NULL, invoke_count INTEGER NOT NULL DEFAULT 1,
              receipt TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS operations (
              workflow_id TEXT PRIMARY KEY, idempotency_key TEXT UNIQUE NOT NULL,
              issue_version INTEGER, causal_parent TEXT, state TEXT NOT NULL,
              receipt TEXT, receipt_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS evidence (
              sequence_no INTEGER PRIMARY KEY AUTOINCREMENT, workflow_id TEXT NOT NULL,
              kind TEXT NOT NULL, payload TEXT NOT NULL, previous_hash TEXT NOT NULL, event_hash TEXT NOT NULL
            );
            """
        )
        self._add_column_if_missing("external_sink", "invoke_count", "INTEGER NOT NULL DEFAULT 1")
        self._add_column_if_missing("external_sink", "workflow_id", "TEXT")
        self._add_column_if_missing("external_sink", "workflow_version", "INTEGER")
        self._add_column_if_missing("external_sink", "causal_parent", "TEXT")
        self._add_column_if_missing("operations", "issue_version", "INTEGER")
        self._add_column_if_missing("operations", "causal_parent", "TEXT")
        self._add_column_if_missing("operations", "receipt_count", "INTEGER NOT NULL DEFAULT 0")
        self.connection.commit()

    def _add_column_if_missing(self, table: str, column: str, definition: str) -> None:
        columns = {row["name"] for row in self.connection.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _append_evidence(self, workflow_id: str, kind: str, payload: dict[str, Any]) -> None:
        row = self.connection.execute(
            "SELECT event_hash FROM evidence WHERE workflow_id = ? ORDER BY sequence_no DESC LIMIT 1",
            (workflow_id,),
        ).fetchone()
        previous = row["event_hash"] if row else "GENESIS"
        self.connection.execute(
            "INSERT INTO evidence(workflow_id, kind, payload, previous_hash, event_hash) VALUES(?, ?, ?, ?, ?)",
            (
                workflow_id,
                kind,
                json.dumps(payload, sort_keys=True),
                previous,
                _hash(previous, workflow_id, kind, payload),
            ),
        )

    def _evidence_head(self, workflow_id: str) -> str:
        row = self.connection.execute(
            "SELECT event_hash FROM evidence WHERE workflow_id = ? ORDER BY sequence_no DESC LIMIT 1",
            (workflow_id,),
        ).fetchone()
        return row["event_hash"] if row else "GENESIS"

    @staticmethod
    def _expected_idempotency_key(workflow_id: str) -> str:
        digest = hmac.new(
            _ISSUE_AUTHENTICATION_KEY,
            f"{workflow_id}|dispatch|v1".encode(),
            hashlib.sha256,
        ).hexdigest()
        return f"skyrescue:{digest}"

    @staticmethod
    def _receipt_claims(
        idempotency_key: str, workflow_version: int, causal_parent: str
    ) -> dict[str, Any]:
        return {
            "causal_parent": causal_parent,
            "idempotency_key": idempotency_key,
            "outcome": "effect_occurred",
            "workflow_issue_version": workflow_version,
        }

    @classmethod
    def _make_receiver_receipt(
        cls, idempotency_key: str, workflow_version: int, causal_parent: str
    ) -> str:
        claims = cls._receipt_claims(idempotency_key, workflow_version, causal_parent)
        encoded = json.dumps(claims, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        claims["authenticator"] = hmac.new(
            _RECEIVER_AUTHENTICATION_KEY, encoded.encode(), hashlib.sha256
        ).hexdigest()
        return json.dumps(claims, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

    @classmethod
    def _receipt_is_valid(
        cls,
        receipt: str | None,
        idempotency_key: str,
        workflow_version: int,
        causal_parent: str,
    ) -> bool:
        if not isinstance(receipt, str):
            return False
        try:
            parsed = json.loads(receipt)
        except (json.JSONDecodeError, TypeError):
            return False
        if not isinstance(parsed, dict) or set(parsed) != {
            "authenticator",
            "causal_parent",
            "idempotency_key",
            "outcome",
            "workflow_issue_version",
        }:
            return False
        authenticator = parsed.pop("authenticator")
        if not isinstance(authenticator, str):
            return False
        expected_claims = cls._receipt_claims(idempotency_key, workflow_version, causal_parent)
        if parsed != expected_claims:
            return False
        encoded = json.dumps(parsed, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        expected = hmac.new(
            _RECEIVER_AUTHENTICATION_KEY, encoded.encode(), hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(authenticator, expected)

    def start(self, workflow_id: str) -> None:
        inserted = self.connection.execute(
            "INSERT OR IGNORE INTO workflows(workflow_id, version, status) VALUES(?, 1, 'Proposed')",
            (workflow_id,),
        )
        self.connection.execute(
            "INSERT OR IGNORE INTO reservations(workflow_id, slot, state) VALUES(?, 'Zone-SE-07@T0', 'Reserved')",
            (workflow_id,),
        )
        if inserted.rowcount:
            self._append_evidence(workflow_id, "proposal", {"workflow_id": workflow_id})
        self.connection.commit()

    def _ensure_operation(self, workflow_id: str) -> sqlite3.Row:
        key = self._expected_idempotency_key(workflow_id)
        self.connection.execute(
            "INSERT OR IGNORE INTO operations(workflow_id, idempotency_key, state) VALUES(?, ?, 'Proposed')",
            (workflow_id, key),
        )
        self.connection.commit()
        return self.connection.execute(
            "SELECT * FROM operations WHERE workflow_id = ?", (workflow_id,)
        ).fetchone()

    def _precheck(self, workflow_id: str) -> sqlite3.Row:
        operation = self._operation(workflow_id)
        if not isinstance(operation["idempotency_key"], str) or not hmac.compare_digest(
            operation["idempotency_key"], self._expected_idempotency_key(workflow_id)
        ):
            self._fail_safe(workflow_id, "idempotency_key_mismatch", "issue_verification_failed")
            raise RuntimeError("Operation identity failed verification")
        updated = self.connection.execute(
            "UPDATE operations SET state = ? WHERE workflow_id = ? AND state = ?",
            (OperationState.PRECHECKED.value, workflow_id, OperationState.PROPOSED.value),
        )
        if updated.rowcount != 1:
            raise RuntimeError(f"Cannot precheck operation from {operation['state']}")
        self._append_evidence(workflow_id, "precheck_passed", {"workflow_id": workflow_id})
        self.connection.commit()
        return self._operation(workflow_id)

    def _mark_executing(self, workflow_id: str) -> sqlite3.Row:
        operation = self._operation(workflow_id)
        if not isinstance(operation["idempotency_key"], str) or not hmac.compare_digest(
            operation["idempotency_key"], self._expected_idempotency_key(workflow_id)
        ):
            self._fail_safe(workflow_id, "idempotency_key_mismatch", "issue_verification_failed")
            raise RuntimeError("Operation identity failed verification")
        workflow = self.connection.execute(
            "SELECT version FROM workflows WHERE workflow_id = ?", (workflow_id,)
        ).fetchone()
        if workflow is None:
            raise KeyError(f"Unknown workflow: {workflow_id}")
        issue_version = int(workflow["version"])
        causal_parent = self._evidence_head(workflow_id)
        updated = self.connection.execute(
            """UPDATE operations
               SET state = ?, issue_version = ?, causal_parent = ?, receipt = NULL, receipt_count = 0
               WHERE workflow_id = ? AND state = ?""",
            (
                OperationState.EXECUTING.value,
                issue_version,
                causal_parent,
                workflow_id,
                OperationState.PRECHECKED.value,
            ),
        )
        if updated.rowcount != 1:
            raise RuntimeError(f"Cannot prepare issue from {operation['state']}")
        self._append_evidence(
            workflow_id,
            "issue_prepared",
            {
                "causal_parent": causal_parent,
                "idempotency_key": operation["idempotency_key"],
                "workflow_issue_version": issue_version,
            },
        )
        self.connection.commit()
        return self._operation(workflow_id)

    def _operation(self, workflow_id: str) -> sqlite3.Row:
        operation = self.connection.execute(
            "SELECT * FROM operations WHERE workflow_id = ?", (workflow_id,)
        ).fetchone()
        if operation is None:
            raise KeyError(f"Unknown workflow operation: {workflow_id}")
        return operation

    def _issue_record_is_valid(self, workflow_id: str, operation: sqlite3.Row) -> bool:
        key = operation["idempotency_key"]
        issue_version = operation["issue_version"]
        causal_parent = operation["causal_parent"]
        if (
            not isinstance(key, str)
            or not hmac.compare_digest(key, self._expected_idempotency_key(workflow_id))
            or type(issue_version) is not int
            or not isinstance(causal_parent, str)
            or causal_parent == "GENESIS"
        ):
            return False
        evidence_rows = self.connection.execute(
            """SELECT payload, previous_hash FROM evidence
               WHERE workflow_id = ? AND kind = 'issue_prepared'
               ORDER BY sequence_no DESC""",
            (workflow_id,),
        ).fetchall()
        expected_payload = {
            "causal_parent": causal_parent,
            "idempotency_key": key,
            "workflow_issue_version": issue_version,
        }
        matching_issue = False
        for row in evidence_rows:
            try:
                payload = json.loads(row["payload"])
            except (json.JSONDecodeError, TypeError):
                continue
            if row["previous_hash"] == causal_parent and payload == expected_payload:
                matching_issue = True
                break
        if not matching_issue:
            return False
        parent = self.connection.execute(
            "SELECT 1 FROM evidence WHERE workflow_id = ? AND event_hash = ?",
            (workflow_id, causal_parent),
        ).fetchone()
        return parent is not None

    def _receiver_row_is_valid(self, operation: sqlite3.Row, sink: sqlite3.Row) -> bool:
        return bool(
            sink["idempotency_key"] == operation["idempotency_key"]
            and sink["workflow_id"] == operation["workflow_id"]
            and sink["workflow_version"] == operation["issue_version"]
            and sink["causal_parent"] == operation["causal_parent"]
            and sink["effect_count"] == 1
            and self._receipt_is_valid(
                sink["receipt"],
                operation["idempotency_key"],
                operation["issue_version"],
                operation["causal_parent"],
            )
        )

    def _fail_safe(self, workflow_id: str, reason: str, evidence_kind: str) -> str:
        self.connection.execute(
            """UPDATE operations
               SET state = ?, receipt = NULL, receipt_count = 0
               WHERE workflow_id = ?""",
            (OperationState.HUMAN_ESCALATED.value, workflow_id),
        )
        self.connection.execute(
            "UPDATE workflows SET status = ? WHERE workflow_id = ?",
            (OperationState.HUMAN_ESCALATED.value, workflow_id),
        )
        self._append_evidence(workflow_id, evidence_kind, {"reason": reason})
        self.connection.commit()
        return OperationState.HUMAN_ESCALATED.value

    def _invoke_external_effect(self, idempotency_key: str) -> str:
        operation = self.connection.execute(
            "SELECT * FROM operations WHERE idempotency_key = ?", (idempotency_key,)
        ).fetchone()
        if operation is None or not self._issue_record_is_valid(operation["workflow_id"], operation):
            raise RuntimeError("Receiver invocation lacks a valid durable issue record")
        if operation["state"] != OperationState.EXECUTING.value:
            raise RuntimeError(f"Cannot invoke receiver from {operation['state']}")

        sink = self.connection.execute(
            "SELECT * FROM external_sink WHERE idempotency_key = ?", (idempotency_key,)
        ).fetchone()
        if sink is not None:
            if not self._receiver_row_is_valid(operation, sink):
                raise RuntimeError("Receiver detected a conflicting or invalid operation binding")
            self.connection.execute(
                "UPDATE external_sink SET invoke_count = invoke_count + 1 WHERE idempotency_key = ?",
                (idempotency_key,),
            )
            self._append_evidence(
                operation["workflow_id"],
                "receiver_duplicate_suppressed",
                {
                    "causal_parent": operation["causal_parent"],
                    "idempotency_key": idempotency_key,
                    "workflow_issue_version": operation["issue_version"],
                },
            )
            self.connection.commit()
            return sink["receipt"]

        receipt = self._make_receiver_receipt(
            idempotency_key, operation["issue_version"], operation["causal_parent"]
        )
        self.connection.execute(
            """INSERT INTO external_sink(
                 idempotency_key, workflow_id, workflow_version, causal_parent,
                 effect_count, invoke_count, receipt
               ) VALUES(?, ?, ?, ?, 1, 1, ?)""",
            (
                idempotency_key,
                operation["workflow_id"],
                operation["issue_version"],
                operation["causal_parent"],
                receipt,
            ),
        )
        self._append_evidence(
            operation["workflow_id"],
            "receiver_effect_applied",
            {
                "causal_parent": operation["causal_parent"],
                "idempotency_key": idempotency_key,
                "receipt": receipt,
                "workflow_issue_version": operation["issue_version"],
            },
        )
        self.connection.commit()
        return receipt

    def _persist_receipt(self, workflow_id: str, receipt: str, evidence_kind: str) -> bool:
        operation = self._operation(workflow_id)
        if operation["state"] not in {
            OperationState.EXECUTING.value,
            OperationState.EFFECT_UNKNOWN.value,
        } or not self._issue_record_is_valid(workflow_id, operation):
            self._fail_safe(workflow_id, "invalid_durable_issue", "receipt_verification_failed")
            return False
        sink = self.connection.execute(
            "SELECT * FROM external_sink WHERE idempotency_key = ?", (operation["idempotency_key"],)
        ).fetchone()
        if (
            sink is None
            or not self._receiver_row_is_valid(operation, sink)
            or receipt != sink["receipt"]
            or not self._receipt_is_valid(
                receipt,
                operation["idempotency_key"],
                operation["issue_version"],
                operation["causal_parent"],
            )
        ):
            self._fail_safe(workflow_id, "invalid_receiver_receipt", "receipt_verification_failed")
            return False
        self.connection.execute(
            """UPDATE operations
               SET state = ?, receipt = ?, receipt_count = 1
               WHERE workflow_id = ? AND state != ?""",
            (OperationState.COMMITTED.value, receipt, workflow_id, OperationState.COMMITTED.value),
        )
        self.connection.execute(
            "UPDATE workflows SET version = ?, status = ? WHERE workflow_id = ?",
            (operation["issue_version"] + 1, OperationState.COMMITTED.value, workflow_id),
        )
        self._append_evidence(
            workflow_id,
            evidence_kind,
            {
                "causal_parent": operation["causal_parent"],
                "idempotency_key": operation["idempotency_key"],
                "receipt": receipt,
                "workflow_issue_version": operation["issue_version"],
            },
        )
        self.connection.commit()
        return True

    def _committed_state_is_valid(self, workflow_id: str, operation: sqlite3.Row) -> bool:
        workflow = self.connection.execute(
            "SELECT * FROM workflows WHERE workflow_id = ?", (workflow_id,)
        ).fetchone()
        sink = self.connection.execute(
            "SELECT * FROM external_sink WHERE idempotency_key = ?", (operation["idempotency_key"],)
        ).fetchone()
        return bool(
            workflow is not None
            and self._issue_record_is_valid(workflow_id, operation)
            and operation["receipt_count"] == 1
            and sink is not None
            and self._receiver_row_is_valid(operation, sink)
            and operation["receipt"] == sink["receipt"]
            and workflow["status"] == OperationState.COMMITTED.value
            and workflow["version"] == operation["issue_version"] + 1
        )

    def reconcile(self, workflow_id: str, receiver_query_available: bool = True) -> str:
        """Resolve an issued call whose externally visible outcome is uncertain.

        If the receiver confirms the effect, persist its receipt.  If it confirms
        absence, return to ``Prechecked`` so a normal retry may be issued.  If the
        receiver cannot be queried, stop in ``HumanEscalated`` rather than guessing.
        """
        operation = self._ensure_operation(workflow_id)
        if operation["state"] == OperationState.COMMITTED.value:
            if self._committed_state_is_valid(workflow_id, operation):
                return OperationState.COMMITTED.value
            return self._fail_safe(
                workflow_id, "invalid_committed_receipt", "receipt_verification_failed"
            )
        if operation["state"] in {
            OperationState.EXECUTING.value,
            OperationState.EFFECT_UNKNOWN.value,
        } and not self._issue_record_is_valid(workflow_id, operation):
            return self._fail_safe(
                workflow_id, "invalid_durable_issue", "issue_verification_failed"
            )
        if operation["state"] == OperationState.EXECUTING.value:
            self.connection.execute(
                "UPDATE operations SET state = ? WHERE workflow_id = ?",
                (OperationState.EFFECT_UNKNOWN.value, workflow_id),
            )
            self._append_evidence(workflow_id, "effect_outcome_unknown", {"workflow_id": workflow_id})
            self.connection.commit()
            operation = self._operation(workflow_id)
        if operation["state"] != OperationState.EFFECT_UNKNOWN.value:
            return operation["state"]
        if not receiver_query_available:
            self.connection.execute(
                "UPDATE operations SET state = ? WHERE workflow_id = ?",
                (OperationState.HUMAN_ESCALATED.value, workflow_id),
            )
            self.connection.execute(
                "UPDATE workflows SET status = ? WHERE workflow_id = ?",
                (OperationState.HUMAN_ESCALATED.value, workflow_id),
            )
            self._append_evidence(workflow_id, "reconciliation_unavailable", {"workflow_id": workflow_id})
            self.connection.commit()
            return OperationState.HUMAN_ESCALATED.value
        sink = self.connection.execute(
            "SELECT * FROM external_sink WHERE idempotency_key = ?", (operation["idempotency_key"],)
        ).fetchone()
        if sink is None:
            self.connection.execute(
                "UPDATE operations SET state = ? WHERE workflow_id = ?",
                (OperationState.PRECHECKED.value, workflow_id),
            )
            self._append_evidence(workflow_id, "reconciled_effect_absent", {"workflow_id": workflow_id})
            self.connection.commit()
            return OperationState.PRECHECKED.value
        if not self._receiver_row_is_valid(operation, sink):
            return self._fail_safe(
                workflow_id, "invalid_receiver_receipt", "receipt_verification_failed"
            )
        receipt = sink["receipt"]
        if not self._persist_receipt(workflow_id, receipt, "reconciled_effect_occurred"):
            return OperationState.HUMAN_ESCALATED.value
        return OperationState.COMMITTED.value

    @staticmethod
    def _coerce_crash_point(crash_point: CrashPoint | str | None) -> CrashPoint | None:
        if crash_point is None or isinstance(crash_point, CrashPoint):
            return crash_point
        return CrashPoint(crash_point)

    def _crash(self) -> None:
        self.connection.close()
        os._exit(CRASH_EXIT_CODE)

    def execute(
        self,
        workflow_id: str,
        crash_point: CrashPoint | str | None = None,
        *,
        crash_after_effect: bool = False,
        receiver_query_available: bool = True,
    ) -> None:
        crash_point = self._coerce_crash_point(
            CrashPoint.AFTER_EFFECT_BEFORE_RECEIPT if crash_after_effect else crash_point
        )
        operation = self._ensure_operation(workflow_id)
        if operation["state"] == OperationState.PROPOSED.value:
            operation = self._precheck(workflow_id)
        if operation["state"] == OperationState.COMMITTED.value:
            if self._committed_state_is_valid(workflow_id, operation):
                return
            self._fail_safe(workflow_id, "invalid_committed_receipt", "receipt_verification_failed")
            raise RuntimeError("Committed receipt failed verification; human adjudication is required")
        if operation["state"] in {OperationState.EXECUTING.value, OperationState.EFFECT_UNKNOWN.value}:
            state = self.reconcile(workflow_id, receiver_query_available=receiver_query_available)
            if state == OperationState.COMMITTED.value:
                return
            if state == OperationState.HUMAN_ESCALATED.value:
                raise RuntimeError("Receiver outcome is unknown; human adjudication is required")
            operation = self._operation(workflow_id)
        if operation["state"] == OperationState.HUMAN_ESCALATED.value:
            raise RuntimeError("Workflow is awaiting human adjudication")
        if operation["state"] != OperationState.PRECHECKED.value:
            raise RuntimeError(f"Cannot issue external call from {operation['state']}")
        operation = self._mark_executing(workflow_id)
        if crash_point == CrashPoint.BEFORE_EXTERNAL_CALL:
            self._crash()
        try:
            receipt = self._invoke_external_effect(operation["idempotency_key"])
        except RuntimeError as error:
            self._fail_safe(workflow_id, "receiver_binding_conflict", "receipt_verification_failed")
            raise RuntimeError("Receiver binding failed verification; human adjudication is required") from error
        if crash_point == CrashPoint.AFTER_EFFECT_BEFORE_RECEIPT:
            self._crash()
        if not self._persist_receipt(workflow_id, receipt, "receipt_persisted"):
            raise RuntimeError("Receiver receipt failed verification; human adjudication is required")
        if crash_point == CrashPoint.AFTER_RECEIPT_PERSISTED:
            self._crash()

    def inspect(self, workflow_id: str) -> dict[str, Any]:
        workflow = self.connection.execute(
            "SELECT * FROM workflows WHERE workflow_id = ?", (workflow_id,)
        ).fetchone()
        operation = self._operation(workflow_id)
        sink = self.connection.execute(
            "SELECT * FROM external_sink WHERE idempotency_key = ?", (operation["idempotency_key"],)
        ).fetchone()
        reservations = self.connection.execute(
            "SELECT * FROM reservations WHERE workflow_id = ?", (workflow_id,)
        ).fetchall()
        evidence = self.connection.execute(
            "SELECT * FROM evidence WHERE workflow_id = ? ORDER BY sequence_no", (workflow_id,)
        ).fetchall()
        previous = "GENESIS"
        chain_valid = True
        evidence_kinds: list[str] = []
        for entry in evidence:
            payload = json.loads(entry["payload"])
            chain_valid = (
                chain_valid
                and entry["previous_hash"] == previous
                and entry["event_hash"]
                == _hash(previous, entry["workflow_id"], entry["kind"], payload)
            )
            previous = entry["event_hash"]
            evidence_kinds.append(entry["kind"])
        receiver_receipt_valid = bool(
            sink is not None
            and operation["issue_version"] is not None
            and operation["causal_parent"] is not None
            and self._receiver_row_is_valid(operation, sink)
        )
        local_receipt_valid = bool(
            operation["receipt_count"] == 1
            and receiver_receipt_valid
            and operation["receipt"] == sink["receipt"]
        )
        return {
            "workflow_status": workflow["status"],
            "workflow_version": workflow["version"],
            "operation_state": operation["state"],
            "idempotency_key": operation["idempotency_key"],
            "issue_version": operation["issue_version"],
            "causal_parent": operation["causal_parent"],
            "invoke_count": sink["invoke_count"] if sink else 0,
            "effect_count": sink["effect_count"] if sink else 0,
            "receipt_count": operation["receipt_count"],
            "receiver_receipt_valid": receiver_receipt_valid,
            "local_receipt_valid": local_receipt_valid,
            "reservation_count": len(reservations),
            "reservation_consistent": len(reservations) == 1 and reservations[0]["state"] == "Reserved",
            "evidence_chain_continuous": chain_valid,
            "evidence_kinds": evidence_kinds,
        }
