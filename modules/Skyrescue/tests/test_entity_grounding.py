import inspect

from skyrescue.entity_grounding import (
    anchors_equivalent,
    compile_grounded_candidate,
    ground_target,
)


def valid_candidate(target_zone: str) -> dict[str, str]:
    return {
        "task_type": "SearchAndRescue",
        "target_zone": target_zone,
        "priority": "Critical",
        "deadline_s_or_text": "urgent_unspecified",
        "required_skill": "thermal_recon",
        "needs_human_approval": "No",
        "expected_failure": "None",
    }


def test_online_grounder_has_no_gold_label_parameter():
    assert tuple(inspect.signature(ground_target).parameters) == (
        "target_text",
        "scenario_card",
        "instruction_text",
    )


def test_aliases_resolve_to_same_frozen_anchor():
    scenario = "山区地震后有村落失联，需要对指定山谷开展热成像搜索。"
    instruction = "失联村子那条山谷，派带热成像的机子过去扫一遍。"
    predicted = ground_target("失联村子所在山谷", scenario, instruction)
    gold = ground_target("失联村落所在山谷", scenario, instruction)
    assert predicted.resolved and gold.resolved
    assert predicted.anchor_ids == ("valley", "village")
    assert predicted.anchor_ids == gold.anchor_ids
    assert anchors_equivalent(predicted, gold)


def test_generic_target_is_not_forced_when_context_has_multiple_entities():
    anchor = ground_target(
        "那里",
        "桥梁坍塌后，医院和桥面均需要检查。",
        "先去那里看一下，等值班员确认后再继续。",
    )
    assert not anchor.resolved
    assert anchor.reason in {"ambiguous_generic_target", "no_context_entity"}


def test_explicit_unresolved_marker_remains_unresolved():
    anchor = ground_target(
        "Unspecified/needs_expert_grounding",
        "多处设施受损。",
        "安排一架过去看看。",
    )
    assert not anchor.resolved
    assert anchor.reason == "explicit_unresolved"


def test_resolved_candidate_receives_anchor_id():
    result = compile_grounded_candidate(
        valid_candidate("失联村子所在山谷"),
        "山区地震后有村落失联，需要对指定山谷开展热成像搜索。",
        "失联村子那条山谷，派带热成像的机子过去扫一遍。",
    )
    assert result.compilation.executable
    assert result.compilation.tasks[0]["target_anchor_ids"] == ["valley", "village"]


def test_unresolved_candidate_is_stopped_before_execution():
    result = compile_grounded_candidate(
        valid_candidate("那里"),
        "桥梁坍塌后，医院和桥面均需要检查。",
        "先去那里看一下。",
    )
    assert not result.compilation.executable
    assert result.compilation.failure == "UngroundedEntity"
