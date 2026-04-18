from __future__ import annotations

from skyshield.decision.threat import score_threat


def test_low_altitude_intruder_scores_high(default_cfg):
    # A target cruising at 130 m directly toward a no-fly zone.
    zone = default_cfg.city.no_fly_zones[0]
    pos_m = (zone.center_km[0] * 1000.0 - 1200.0,
             zone.center_km[1] * 1000.0, 130.0)
    vel = (30.0, 0.0, 0.0)
    s = score_threat(pos_m, vel, default_cfg.city, target_class_conf=0.9)
    assert s >= 0.55


def test_far_slow_benign_scores_low(default_cfg):
    pos_m = (19000.0, 14500.0, 300.0)
    vel = (2.0, 0.0, 0.0)
    s = score_threat(pos_m, vel, default_cfg.city, target_class_conf=0.5)
    assert s < 0.55
