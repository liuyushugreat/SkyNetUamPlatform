"""SQLite-backed crash-recovery prototype for one non-idempotent workflow action."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any


CRASH_EXIT_CODE = 70


def _hash(previous: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{previous}|{encoded}".encode()).hexdigest()


class DurableWorkflowRuntime:
    """Persist proposal/commit state and reconcile a simulated external sink."""

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
              idempotency_key TEXT PRIMARY KEY, effect_count INTEGER NOT NULL, receipt TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS operations (
              workflow_id TEXT PRIMARY KEY, idempotency_key TEXT UNIQUE NOT NULL,
              state TEXT NOT NULL, receipt TEXT
            );
            CREATE TABLE IF NOT EXISTS evidence (
              sequence_no INTEGER PRIMARY KEY AUTOINCREMENT, workflow_id TEXT NOT NULL,
              kind TEXT NOT NULL, payload TEXT NOT NULL, previous_hash TEXT NOT NULL, event_hash TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def _append_evidence(self, workflow_id: str, kind: str, payload: dict[str, Any]) -> None:
        row = self.connection.execute(
            "SELECT event_hash FROM evidence WHERE workflow_id = ? ORDER BY sequence_no DESC LIMIT 1",
            (workflow_id,),
        ).fetchone()
        previous = row["event_hash"] if row else "GENESIS"
        self.connection.execute(
            "INSERT INTO evidence(workflow_id, kind, payload, previous_hash, event_hash) VALUES(?, ?, ?, ?, ?)",
            (workflow_id, kind, json.dumps(payload, sort_keys=True), previous, _hash(previous, payload)),
        )

    def start(self, workflow_id: str) -> None:
        self.connection.execute(
            "INSERT OR IGNORE INTO workflows(workflow_id, version, status) VALUES(?, 1, 'Prepared')",
            (workflow_id,),
        )
        self.connection.execute(
            "INSERT OR IGNORE INTO reservations(workflow_id, slot, state) VALUES(?, 'Zone-SE-07@T0', 'Reserved')",
            (workflow_id,),
        )
        self._append_evidence(workflow_id, "proposal", {"workflow_id": workflow_id})
        self.connection.commit()

    def _ensure_operation(self, workflow_id: str) -> sqlite3.Row:
        key = f"{workflow_id}:dispatch:v1"
        self.connection.execute(
            "INSERT OR IGNORE INTO operations(workflow_id, idempotency_key, state) VALUES(?, ?, 'Proposed')",
            (workflow_id, key),
        )
        self.connection.commit()
        return self.connection.execute(
            "SELECT * FROM operations WHERE workflow_id = ?", (workflow_id,)
        ).fetchone()

    def _external_effect(self, idempotency_key: str) -> str:
        receipt = hashlib.sha256(idempotency_key.encode()).hexdigest()[:16]
        self.connection.execute(
            "INSERT OR IGNORE INTO external_sink(idempotency_key, effect_count, receipt) VALUES(?, 1, ?)",
            (idempotency_key, receipt),
        )
        self.connection.commit()
        return receipt

    def _reconcile_sink(self, workflow_id: str, operation: sqlite3.Row) -> str | None:
        sink = self.connection.execute(
            "SELECT receipt FROM external_sink WHERE idempotency_key = ?", (operation["idempotency_key"],)
        ).fetchone()
        if sink is None:
            return None
        receipt = sink["receipt"]
        self.connection.execute(
            "UPDATE operations SET state = 'Committed', receipt = ? WHERE workflow_id = ?",
            (receipt, workflow_id),
        )
        self.connection.execute(
            "UPDATE workflows SET version = 2, status = 'Committed' WHERE workflow_id = ?", (workflow_id,)
        )
        self._append_evidence(workflow_id, "reconciled_commit", {"receipt": receipt})
        self.connection.commit()
        return receipt

    def execute(self, workflow_id: str, crash_after_effect: bool = False) -> None:
        operation = self._ensure_operation(workflow_id)
        if operation["state"] == "Committed":
            return
        if self._reconcile_sink(workflow_id, operation) is not None:
            return
        receipt = self._external_effect(operation["idempotency_key"])
        if crash_after_effect:
            self.connection.close()
            os._exit(CRASH_EXIT_CODE)
        self.connection.execute(
            "UPDATE operations SET state = 'Committed', receipt = ? WHERE workflow_id = ?",
            (receipt, workflow_id),
        )
        self.connection.execute(
            "UPDATE workflows SET version = 2, status = 'Committed' WHERE workflow_id = ?", (workflow_id,)
        )
        self._append_evidence(workflow_id, "commit", {"receipt": receipt})
        self.connection.commit()

    def inspect(self, workflow_id: str) -> dict[str, Any]:
        workflow = self.connection.execute(
            "SELECT * FROM workflows WHERE workflow_id = ?", (workflow_id,)
        ).fetchone()
        operation = self.connection.execute(
            "SELECT * FROM operations WHERE workflow_id = ?", (workflow_id,)
        ).fetchone()
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
        for entry in evidence:
            payload = json.loads(entry["payload"])
            chain_valid = chain_valid and entry["previous_hash"] == previous and entry["event_hash"] == _hash(previous, payload)
            previous = entry["event_hash"]
        return {
            "workflow_status": workflow["status"],
            "workflow_version": workflow["version"],
            "operation_state": operation["state"],
            "effect_count": sink["effect_count"],
            "reservation_count": len(reservations),
            "reservation_consistent": len(reservations) == 1 and reservations[0]["state"] == "Reserved",
            "evidence_chain_continuous": chain_valid,
        }
