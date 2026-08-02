"""Deterministic authorization boundary used by SkyRescue security evaluation."""

from __future__ import annotations


ACTION_PERMISSIONS = {
    "reserve_airspace": "airspace.reserve",
    "dispatch_uav": "mission.dispatch",
    "return_to_base": "mission.recover",
}


def evaluate_action(request: dict, seen_idempotency_keys: set[str]) -> tuple[bool, str]:
    """Return an allow/deny decision using only runtime request fields."""
    action = request.get("action")
    if action not in ACTION_PERMISSIONS:
        return False, "unknown_action"
    if request.get("source_trust") != "authenticated":
        return False, "untrusted_source"
    if request.get("permission") != ACTION_PERMISSIONS[action]:
        return False, "permission_mismatch"
    if request.get("risk") == "high" and not request.get("human_approval"):
        return False, "approval_required"
    key = request.get("idempotency_key")
    if not isinstance(key, str) or not key or key in seen_idempotency_keys:
        return False, "replay_or_missing_key"
    if set(request) - {"request_id", "action", "permission", "source_trust", "risk", "human_approval", "idempotency_key", "operator_text"}:
        return False, "schema_violation"
    text = str(request.get("operator_text", "")).lower()
    if any(marker in text for marker in ("ignore previous", "system override", "bypass approval")):
        return False, "prompt_injection"
    seen_idempotency_keys.add(key)
    return True, "allowed"
