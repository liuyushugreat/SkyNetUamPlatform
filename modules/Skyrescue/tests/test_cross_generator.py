"""Tests for the independent SkyRescue cross-generator challenge."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "generate_cross_generator_challenge.py"


def load_generator_module():
    spec = importlib.util.spec_from_file_location("cross_generator", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smooth_envelope_has_gradual_peak() -> None:
    module = load_generator_module()
    assert module.smooth_envelope(0.0) == 0.0
    assert module.smooth_envelope(0.5) == 1.0
    assert abs(module.smooth_envelope(1.0)) < 1e-12


def test_generator_writes_separated_truth_and_manifest(tmp_path: Path, monkeypatch) -> None:
    module = load_generator_module()
    output = tmp_path / "challenge"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--output",
            str(output),
            "--seed",
            "7",
            "--uavs",
            "4",
            "--duration",
            "600",
            "--faults",
            "12",
        ],
    )
    module.main()

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["name"] == "SkyRescue-CrossGenerator"
    assert manifest["detector_policy"].startswith("all detector thresholds are frozen")
    assert len((output / "faults.jsonl").read_text(encoding="utf-8").splitlines()) == 12
    telemetry_lines = (output / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(telemetry_lines) == 4 * 600
    record = json.loads(telemetry_lines[0])
    assert "fault_type" not in record
    assert {"position_residual_m", "command_latency_ms", "audit_sequence_gap"} <= record.keys()
