"""Shared helpers for the SkyShield experiment scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))


def default_config_path() -> Path:
    return MODULE_ROOT / "configs" / "default.yaml"


def load_field_sorties() -> dict:
    return json.loads(
        (MODULE_ROOT / "data" / "field_sorties.json").read_text(encoding="utf-8")
    )


def load_augmented_seeds() -> dict:
    return json.loads(
        (MODULE_ROOT / "data" / "augmented_seeds.json").read_text(encoding="utf-8")
    )


def real_scenarios():
    """Return the 10 real sorties as ``SortieScenario`` instances."""
    from skyshield.runtime import SortieScenario  # imported here to keep deps lazy

    data = load_field_sorties()
    out = []
    for s in data["sorties"]:
        forced_abort = s["outcome"] == "operator_abort"
        forced_lost = s["outcome"] == "target_lost"
        out.append(
            SortieScenario(
                sortie_id=s["sortie_id"],
                test_type=s["test_type"],
                target_takeoff_t=s["target_takeoff_t"],
                target_speed_kmh=float(s["target_speed_kmh"]),
                target_height_m=float(s["target_height_m"]),
                interceptor_takeoff_t=s["interceptor_takeoff_t"],
                forced_abort=forced_abort,
                forced_lost_lock=forced_lost,
                is_real=True,
                expected_outcome=s["outcome"],
            )
        )
    return out


def augmented_scenarios(rng_seed: int = 20260418):
    """Generate the 50 augmented (replay-extended) sorties."""
    import numpy as np

    from skyshield.runtime import SortieScenario

    aug = load_augmented_seeds()
    real = load_field_sorties()["sorties"]
    rng = np.random.default_rng(rng_seed)
    out = []
    for entry in aug["per_sortie_seed_offset"]:
        tpl = next(s for s in real if s["sortie_id"] == entry["base_template"])
        speed = float(np.clip(tpl["target_speed_kmh"] + rng.normal(0, 6.0), 100, 150))
        height = float(np.clip(tpl["target_height_m"] + rng.normal(0, 8.0), 80, 160))
        # Maneuver g for replay-extended sorties: vary 0.3..2.5 deterministically
        g = float(np.clip(rng.uniform(0.3, 2.5), 0.3, 2.5))
        # Roughly preserve the 1/10 abort rate and 2/10 lost-lock rate.
        u = rng.random()
        forced_abort = u < 0.06
        forced_lost = (not forced_abort) and (rng.random() < 0.12)
        out.append(
            SortieScenario(
                sortie_id=entry["id"],
                test_type=tpl["test_type"],
                target_takeoff_t=tpl["target_takeoff_t"],
                target_speed_kmh=speed,
                target_height_m=height,
                interceptor_takeoff_t=tpl["interceptor_takeoff_t"],
                forced_abort=forced_abort,
                forced_lost_lock=forced_lost,
                target_maneuver_g=g,
                is_real=False,
            )
        )
    return out
