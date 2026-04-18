from __future__ import annotations

from skyshield.decision.safety_guard import SafetyGuard, SafetyDecision


def test_friendly_airspace_triggers_abort(default_cfg):
    sg = SafetyGuard(default_cfg.city, default_cfg.safety)
    zone = default_cfg.city.no_fly_zones[0]
    pos_m = (zone.center_km[0] * 1000.0, zone.center_km[1] * 1000.0, 100.0)
    v = sg.check(pos_m, (0.0, 0.0, 0.0), threat_score=0.8,
                 target_class_conf=0.9, authorized=True)
    assert v.decision is SafetyDecision.ABORT
    assert v.reason == "friendly_airspace"


def test_low_class_confidence_suppressed(default_cfg):
    sg = SafetyGuard(default_cfg.city, default_cfg.safety)
    v = sg.check((6000.0, 6000.0, 120.0), (30.0, 0.0, 0.0),
                 threat_score=0.9, target_class_conf=0.3, authorized=True)
    assert v.decision is SafetyDecision.SUPPRESS


def test_allow_clean(default_cfg):
    sg = SafetyGuard(default_cfg.city, default_cfg.safety)
    v = sg.check((6000.0, 6000.0, 120.0), (30.0, 0.0, 0.0),
                 threat_score=0.8, target_class_conf=0.85, authorized=True)
    assert v.decision is SafetyDecision.ALLOW
