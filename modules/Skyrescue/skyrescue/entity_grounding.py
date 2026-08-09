"""Label-isolated contextual entity grounding for human emergency intents.

The online registry is built exclusively from the scenario card and operator
instruction. Gold ``target_zone`` labels are never accepted by this module and
are used only by offline evaluators after both targets have been resolved.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any

from .workflow import CompilationResult, compile_generated_candidate


# Frozen domain ontology. These are generic operational entity classes rather
# than aliases copied from benchmark labels.
ENTITY_TERMS: dict[str, tuple[str, ...]] = {
    "medical_site": ("医疗点", "医疗站", "救护点", "急救点"),
    "valley": ("山谷", "沟底", "峡谷"),
    "fire_site": ("火场", "火线", "林区", "着火区", "明火"),
    "trail": ("步道", "山路", "游览道"),
    "bridge": ("桥梁", "桥面", "桥塌", "桥"),
    "hazmat_park": ("化工园", "工业园", "化工厂", "厂区"),
    "shelter": ("安置点", "避难点", "避难所"),
    "drop_site": ("投送点", "投放点", "落点"),
    "receiver_site": ("接收点", "交接点", "接收站"),
    "hospital": ("医院", "两院", "院区"),
    "road_segment": ("受损路段", "路段", "道路", "公路"),
    "base_station": ("基站", "通信站", "通信覆盖区"),
    "reservoir": ("水库", "水面", "水域"),
    "navigation_facility": ("导航设施", "导航台", "灯塔"),
    "building": ("建筑", "楼体", "老楼", "房屋"),
    "village": ("村庄", "村落", "村子", "村"),
    "highway_incident": ("高速事故", "高速公路", "高速", "事故区域"),
    "rescue_station": ("救援站", "保障点", "救援点"),
    "communication_tower": ("通信塔", "塔体", "铁塔"),
    "landing_zone": ("起降区", "起降点", "候选场地", "临时起降"),
    "river_segment": ("河段", "河道", "污染带"),
    "island": ("离岛", "孤岛", "岛上"),
    "dam": ("堤坝", "大坝", "坝体"),
    "school": ("学校", "校园", "操场"),
    "community": ("社区", "小区", "居民区"),
    "station": ("场站", "车站", "站点"),
    "warehouse": ("仓库", "物资库", "库房"),
    "port": ("码头", "港口", "泊位"),
    "airspace": ("空域", "航线", "走廊", "禁飞区"),
}

QUALIFIER_TERMS: dict[str, tuple[str, ...]] = {
    "north": ("北侧", "北部", "以北"),
    "south": ("南侧", "南部", "以南"),
    "east": ("东侧", "东部", "以东"),
    "west": ("西侧", "西部", "以西"),
    "upstream": ("上游",),
    "downstream": ("下游",),
    "perimeter": ("外围", "周边", "附近", "外侧"),
    "above": ("上空", "上方", "正上方"),
    "downwind": ("下风向",),
}

UNRESOLVED_MARKERS = (
    "unspecified",
    "needs_expert_grounding",
    "未指定",
    "未知地点",
    "地点不明",
    "待确认",
)

GENERIC_TARGETS = {
    "",
    "那里",
    "那儿",
    "那边",
    "那片区域",
    "该区域",
    "目标区域",
    "指定区域",
    "灾区",
    "现场",
    "附近",
    "外围",
}

OPPOSITE_QUALIFIERS = {
    frozenset(("north", "south")),
    frozenset(("east", "west")),
    frozenset(("upstream", "downstream")),
}


def normalize_entity_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    replacements = {
        "村子": "村落",
        "山里": "山区",
        "上方": "上空",
        "水面": "水域",
        "周边": "外围",
        "附近": "外围",
        "那个": "",
        "那处": "",
        "那条": "",
        "那片": "",
        "一处": "",
        "已确认": "",
        "确认好": "",
        "确认的": "",
        "所在": "",
        "受检": "",
        "当前": "",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def _bigrams(value: str) -> set[str]:
    return {value[index:index + 2] for index in range(max(0, len(value) - 1))}


def _similarity(target: str, context: str) -> float:
    if not target or not context:
        return 0.0
    target_chars = set(target)
    context_chars = set(context)
    coverage = len(target_chars & context_chars) / len(target_chars)
    target_bigrams = _bigrams(target)
    context_bigrams = _bigrams(context)
    dice = (
        2 * len(target_bigrams & context_bigrams) / (len(target_bigrams) + len(context_bigrams))
        if target_bigrams and context_bigrams else 0.0
    )
    sequence = SequenceMatcher(None, target, context).ratio()
    containment = 1.0 if target in context or context in target else 0.0
    return max(containment, 0.50 * coverage + 0.30 * dice + 0.20 * sequence)


def _clauses(scenario_card: str, instruction_text: str) -> list[tuple[str, str]]:
    rows = []
    for source, text in (("scenario", scenario_card), ("instruction", instruction_text)):
        for clause in re.split(r"[，。；、：:！？!?\n—]+", text):
            normalized = normalize_entity_text(clause)
            if len(normalized) >= 2:
                rows.append((source, normalized))
    return rows


def _term_hits(text: str, vocabulary: dict[str, tuple[str, ...]]) -> set[str]:
    hits = set()
    for identifier, aliases in vocabulary.items():
        if any(normalize_entity_text(alias) in text for alias in aliases):
            hits.add(identifier)
    return hits


@dataclass(frozen=True)
class EntityAnchor:
    target_text: str
    normalized_target: str
    resolved: bool
    anchor_ids: tuple[str, ...]
    qualifiers: tuple[str, ...]
    confidence: float
    source: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ground_target(
    target_text: str,
    scenario_card: str,
    instruction_text: str,
) -> EntityAnchor:
    """Resolve one target using only the online scenario and instruction text."""

    normalized = normalize_entity_text(target_text)
    marker_text = str(target_text or "").lower()
    if any(marker in marker_text for marker in UNRESOLVED_MARKERS):
        return EntityAnchor(str(target_text), normalized, False, (), (), 1.0, "marker", "explicit_unresolved")

    explicit = _term_hits(normalized, ENTITY_TERMS)
    qualifiers = _term_hits(normalized, QUALIFIER_TERMS)
    if explicit:
        return EntityAnchor(
            str(target_text), normalized, True, tuple(sorted(explicit)), tuple(sorted(qualifiers)),
            1.0, "target", "explicit_domain_entity",
        )

    if normalized in {normalize_entity_text(value) for value in GENERIC_TARGETS}:
        generic = True
    else:
        generic = len(normalized) < 2

    scored: dict[str, tuple[float, str]] = {}
    for source, clause in _clauses(scenario_card, instruction_text):
        score = _similarity(normalized, clause)
        for anchor_id in _term_hits(clause, ENTITY_TERMS):
            previous = scored.get(anchor_id, (0.0, source))
            if score > previous[0]:
                scored[anchor_id] = (score, source)

    ranked = sorted(scored.items(), key=lambda item: (-item[1][0], item[0]))
    if not ranked:
        return EntityAnchor(str(target_text), normalized, False, (), tuple(sorted(qualifiers)), 0.0, "none", "no_context_entity")

    best_id, (best_score, best_source) = ranked[0]
    second_score = ranked[1][1][0] if len(ranked) > 1 else 0.0
    if generic and len(ranked) > 1:
        return EntityAnchor(str(target_text), normalized, False, (), tuple(sorted(qualifiers)), round(best_score, 4), "context", "ambiguous_generic_target")
    if best_score < 0.45 or (len(ranked) > 1 and best_score - second_score < 0.04):
        return EntityAnchor(str(target_text), normalized, False, (), tuple(sorted(qualifiers)), round(best_score, 4), "context", "low_or_ambiguous_similarity")
    return EntityAnchor(
        str(target_text), normalized, True, (best_id,), tuple(sorted(qualifiers)),
        round(best_score, 4), best_source, "contextual_inference",
    )


def anchors_equivalent(left: EntityAnchor, right: EntityAnchor) -> bool:
    """Compare two independently resolved anchors without consulting labels."""

    if not left.resolved or not right.resolved:
        return not left.resolved and not right.resolved and left.reason == right.reason == "explicit_unresolved"
    if not (set(left.anchor_ids) & set(right.anchor_ids)):
        return False
    for opposite in OPPOSITE_QUALIFIERS:
        if opposite <= (set(left.qualifiers) | set(right.qualifiers)):
            if not (opposite <= set(left.qualifiers) or opposite <= set(right.qualifiers)):
                return False
    return True


@dataclass
class GroundedCompilationResult:
    compilation: CompilationResult
    anchor: EntityAnchor

    def to_dict(self) -> dict[str, Any]:
        return {"compilation": self.compilation.to_dict(), "anchor": self.anchor.to_dict()}


def compile_grounded_candidate(
    candidate: dict[str, Any],
    scenario_card: str,
    instruction_text: str,
) -> GroundedCompilationResult:
    """Apply contextual entity grounding before a candidate may execute."""

    base = compile_generated_candidate(candidate)
    anchor = ground_target(str(candidate.get("target_zone", "")), scenario_card, instruction_text)
    if not base.schema_valid or not base.executable or anchor.resolved:
        if base.executable and anchor.resolved:
            tasks = [dict(task, target_anchor_ids=list(anchor.anchor_ids)) for task in base.tasks]
            base = CompilationResult(
                method="skyrescue_grounded_llm_candidate",
                tasks=tasks,
                workflow_nodes=base.workflow_nodes,
                schema_valid=True,
                executable=True,
                failure=None,
                hallucinated_entity=False,
                unregistered_skill_call=False,
                permission_violation=False,
                latency_ms=base.latency_ms,
            )
        return GroundedCompilationResult(base, anchor)

    rejected = CompilationResult(
        method="skyrescue_grounded_llm_candidate",
        tasks=[],
        workflow_nodes=[],
        schema_valid=True,
        executable=False,
        failure="UngroundedEntity",
        hallucinated_entity=True,
        unregistered_skill_call=False,
        permission_violation=False,
        latency_ms=base.latency_ms,
    )
    return GroundedCompilationResult(rejected, anchor)


__all__ = [
    "ENTITY_TERMS",
    "EntityAnchor",
    "GroundedCompilationResult",
    "anchors_equivalent",
    "compile_grounded_candidate",
    "ground_target",
    "normalize_entity_text",
]
