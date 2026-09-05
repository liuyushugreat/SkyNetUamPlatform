from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import pytest

from skyrescue.workflow import build_runtime_event


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_runtime_latency_benchmark.py"
SPEC = importlib.util.spec_from_file_location("run_runtime_latency_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_parser_defaults_to_thirty_measurement_rounds(tmp_path: Path) -> None:
    args = benchmark.build_parser().parse_args([
        "--intent-dataset", str(tmp_path / "intent"),
        "--security-dataset", str(tmp_path / "security"),
        "--output-dir", str(tmp_path / "output"),
    ])
    assert args.warmup_rounds == 5
    assert args.repeats == 30


def test_mechanism_label_names_adjudication_accurately() -> None:
    names = [spec[1] for spec in benchmark.MECHANISM_SPECS]
    assert "Adjudication decision" in names
    assert "Proposal-Adjudication-Commit" not in names


def test_summary_keeps_stack_dimensions_and_all_required_statistics() -> None:
    rows = [
        {
            "benchmark_family": "configured_stack",
            "mechanism": "Configured-stack local repair",
            "framework": "Native",
            "persistence": "off",
            "case_id": "I0001",
            "repeat": repeat,
            "latency_ms": latency,
        }
        for repeat, latency in enumerate((1.0, 2.0, 3.0, 4.0), start=1)
    ]
    summary = benchmark.summarize_rows(
        rows,
        (("configured_stack", "Configured-stack local repair", "Native", "off", 1),),
        repeats=4,
    )[0]
    assert summary["framework"] == "Native"
    assert summary["persistence"] == "off"
    assert summary["p50_ms"] == 2.0
    assert summary["p95_ms"] == 4.0
    assert summary["p99_ms"] == 4.0
    assert summary["mean_ms"] == 2.5
    assert summary["sample_std_ms"] == pytest.approx(1.2909944487)


class FakeGraph:
    def invoke(self, payload, config):
        assert config["configurable"]["thread_id"]
        return {
            "case": payload["case"],
            "event": payload["event"],
            "recovered": {"status": "Recovered", "resources": [{"skills": {"relay", "mapping"}}]},
        }


def _checkpoint_rows(database: Path) -> list[tuple[str, str]]:
    with sqlite3.connect(database) as connection:
        return connection.execute(
            "SELECT boundary, payload_json FROM configured_stack_checkpoints ORDER BY boundary"
        ).fetchall()


def test_native_and_langgraph_use_identical_checkpoint_boundaries(tmp_path: Path, monkeypatch) -> None:
    payload = {
        "case": {"case_id": "I0001"},
        "event": build_runtime_event(0, "node_restart"),
    }
    recovered = {"status": "Recovered", "resources": [{"skills": {"relay", "mapping"}}]}
    monkeypatch.setattr(benchmark, "local_repair", lambda case, profile: recovered)

    native_database = tmp_path / "native.sqlite"
    native_checkpoint = benchmark.MatchedSQLiteCheckpoint(native_database)
    try:
        benchmark.invoke_configured_stack(
            "Native", "on", payload, "run-1", checkpoint=native_checkpoint
        )
    finally:
        native_checkpoint.close()

    graph_database = tmp_path / "langgraph.sqlite"
    graph_checkpoint = benchmark.MatchedSQLiteCheckpoint(graph_database)
    try:
        benchmark.invoke_configured_stack(
            "LangGraph", "on", payload, "run-1", app=FakeGraph(), checkpoint=graph_checkpoint
        )
    finally:
        graph_checkpoint.close()

    native_rows = _checkpoint_rows(native_database)
    graph_rows = _checkpoint_rows(graph_database)
    assert [row[0] for row in native_rows] == ["input", "output"]
    assert native_rows == graph_rows


def test_persistence_off_rejects_a_checkpoint(tmp_path: Path) -> None:
    checkpoint = benchmark.MatchedSQLiteCheckpoint(tmp_path / "unexpected.sqlite")
    try:
        with pytest.raises(ValueError, match="Persistence-on"):
            benchmark.invoke_configured_stack(
                "Native",
                "off",
                {"case": {"case_id": "I0001"}, "event": {}},
                "run-1",
                checkpoint=checkpoint,
            )
    finally:
        checkpoint.close()
