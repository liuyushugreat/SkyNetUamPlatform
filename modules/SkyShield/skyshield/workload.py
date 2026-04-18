"""Deterministic threat-scenario generator.

Two flavours:
  * ``from_field_sorties`` hydrates the 10 real sorties verbatim.
  * ``generate`` samples a Poisson-arrival stream from a config for
    stress / E3 / E4 experiments.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from skyshield.config import SkyShieldConfig
from skyshield.runtime.engine import ThreatScenario


# -------------------------------------------------------------------------- #
# Field-validated scenarios                                                  #
# -------------------------------------------------------------------------- #


def from_field_sorties(
    path: Path,
    cfg: SkyShieldConfig,
    augment: int = 0,
    seed: Optional[int] = None,
) -> List[ThreatScenario]:
    """Read the 10 real sorties and optionally synthesize ``augment`` more.

    Augmented scenarios are strictly labelled as ``replay-extended`` via
    target ids 1000+; no field evidence is fabricated."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    scenarios: List[ThreatScenario] = []
    base_t = 0.0
    for entry in raw["sorties"]:
        scenarios.append(ThreatScenario(
            target_id=int(entry["sortie_id"]),
            appear_ms=base_t + float(entry["appear_s"]) * 1000.0,
            start_pos_m=tuple(entry["start_pos_m"]),
            velocity_mps=tuple(entry["velocity_mps"]),
            target_class_conf=float(entry.get("target_class_conf", 0.85)),
            maneuver=bool(entry.get("maneuver", False)),
            operator_abort=bool(entry.get("operator_abort", False)),
            require_lost=bool(entry.get("target_lost", False)),
        ))
        base_t += 40.0 * 1000.0  # 40 s between real sorties

    if augment > 0:
        rng = np.random.default_rng(seed if seed is not None else cfg.seed)
        bbox = cfg.city.bbox_km
        spd_mu = cfg.workload.target_speed_mean
        spd_sd = cfg.workload.target_speed_std
        alt_mu = cfg.workload.target_altitude_mean
        alt_sd = cfg.workload.target_altitude_std

        for k in range(augment):
            x_km = rng.uniform(bbox[0], bbox[2])
            y_km = rng.uniform(bbox[1], bbox[3])
            alt = max(25.0, rng.normal(alt_mu, alt_sd))
            speed = max(12.0, rng.normal(spd_mu, spd_sd))
            heading = rng.uniform(-math.pi, math.pi)
            vx = speed * math.cos(heading)
            vy = speed * math.sin(heading)
            scenarios.append(ThreatScenario(
                target_id=1000 + k,
                appear_ms=base_t + k * 1800.0,
                start_pos_m=(x_km * 1000.0, y_km * 1000.0, alt),
                velocity_mps=(vx, vy, 0.0),
                target_class_conf=float(np.clip(rng.normal(0.82, 0.08), 0.55, 0.98)),
                maneuver=bool(rng.random() < cfg.workload.maneuver_prob),
                operator_abort=False,
                require_lost=False,
            ))

    return scenarios


# -------------------------------------------------------------------------- #
# Synthetic / replay scenarios                                               #
# -------------------------------------------------------------------------- #


def generate(cfg: SkyShieldConfig, duration_s: Optional[float] = None,
             concurrency: int = 1, seed: Optional[int] = None) -> List[ThreatScenario]:
    rng = np.random.default_rng(seed if seed is not None else cfg.seed)
    duration = duration_s if duration_s is not None else cfg.workload.duration_s
    bbox = cfg.city.bbox_km

    rate = cfg.workload.threat_arrival_rate_hz * max(1, concurrency)
    # Poisson process by thinning: sample inter-arrival times.
    scenarios: List[ThreatScenario] = []
    t = 0.0
    tid = 2000
    while t < duration:
        dt = rng.exponential(1.0 / rate)
        t += dt
        if t >= duration:
            break
        # Concurrency: spawn up to ``concurrency`` simultaneous threats.
        k = 1 + int(rng.integers(0, max(1, concurrency)))
        for _ in range(k):
            x_km = rng.uniform(bbox[0], bbox[2])
            y_km = rng.uniform(bbox[1], bbox[3])
            alt = max(25.0, rng.normal(cfg.workload.target_altitude_mean,
                                       cfg.workload.target_altitude_std))
            speed = max(12.0, rng.normal(cfg.workload.target_speed_mean,
                                         cfg.workload.target_speed_std))
            heading = rng.uniform(-math.pi, math.pi)
            scenarios.append(ThreatScenario(
                target_id=tid,
                appear_ms=t * 1000.0,
                start_pos_m=(x_km * 1000.0, y_km * 1000.0, alt),
                velocity_mps=(speed * math.cos(heading),
                              speed * math.sin(heading), 0.0),
                target_class_conf=float(np.clip(
                    rng.normal(0.82, 0.08), 0.55, 0.98)),
                maneuver=bool(rng.random() < cfg.workload.maneuver_prob),
            ))
            tid += 1

    return scenarios
