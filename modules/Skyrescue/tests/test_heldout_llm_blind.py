"""Tests for the instruction-only held-out LLM capture runner."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_heldout_llm_blind import (  # noqa: E402
    SYSTEM_PROMPT,
    build_user_prompt,
    prompt_sha256,
    validate_cases,
)


def test_prompt_contains_instruction_only_boundary():
    prompt = build_user_prompt({"instruction_id": "HLD-001", "instruction_text": "马上派机去东门搜救"})
    assert prompt == "指挥指令：马上派机去东门搜救"
    assert "场景背景" not in prompt
    assert "仅依据指挥指令" in SYSTEM_PROMPT
    assert len(prompt_sha256()) == 64


def test_blind_input_rejects_extra_fields():
    row = {
        "instruction_id": "HLD-001",
        "instruction_text": "马上派机去东门搜救被困人员",
        "scenario_card": "must not be sent",
    }
    try:
        validate_cases([row])
    except ValueError as exc:
        assert "only instruction_id and instruction_text" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Extra fields must be rejected")


def test_blind_input_accepts_minimal_unique_rows():
    validate_cases([
        {"instruction_id": "HLD-001", "instruction_text": "马上派机去东门搜救被困人员"},
        {"instruction_id": "HLD-002", "instruction_text": "沿河道向北搜索失联巡查人员"},
    ])
