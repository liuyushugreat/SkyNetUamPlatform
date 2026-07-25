"""Declarative scenario specifications for the SkyRwa benchmark.

All parameter distributions, violation injection rules, and governance
expectations are defined here as plain data so that they can be audited,
versioned, and cited independently of the generation logic.
"""

from __future__ import annotations

RANDOM_SEED = 42
BENCHMARK_VERSION = "1.0.0"
BASE_TIME_ISO = "2026-01-15T08:00:00+00:00"

# ---------------------------------------------------------------------------
# Violation taxonomy
# ---------------------------------------------------------------------------
# injected   – deterministically inserted based on flight index (i)
# emergent   – arises from accumulated anomaly scores during pipeline execution
# ---------------------------------------------------------------------------

SCENARIO_SPECS: list[dict] = [
    # ── 1. Clean route survey ───────────────────────────────────────────────
    {
        "tag": "clean_route_survey",
        "count": 12,
        "flight_id_template": "FLT-CLEAN-{i:03d}",
        "uav_id_template": "UAV-A{slot}",
        "uav_slot_modulus": 4,
        "mission_type": "route_survey",
        "asset_class": "ROUTE_OPTIMIZATION_SAMPLE",
        "description": "Routine route surveys with full compliance under ideal conditions.",
        "expected_tradable": True,
        "governance_path": "direct_promotion",
        "injected_violations": [],
        "emergent_violations": [],
        "parameter_distributions": {
            "completed": {"dist": "constant", "value": True},
            "completion_pct": {"dist": "constant", "value": 100.0},
            "telemetry_points": {"dist": "linear", "base": 1000, "step": 80},
            "wind_speed_mps": {"dist": "linear", "base": 2.0, "step": 0.3},
            "visibility_km": {"dist": "constant", "value": 15.0},
            "weather": {"dist": "constant", "value": "clear"},
            "duration_min": {"dist": "linear", "base": 20, "step": 2},
            "anomalies": {"dist": "constant", "value": []},
            "risk_events": {"dist": "constant", "value": []},
            "no_fly_zone_incursions": {"dist": "constant", "value": 0},
        },
    },

    # ── 2. Night flight ─────────────────────────────────────────────────────
    {
        "tag": "night_flight",
        "count": 8,
        "flight_id_template": "FLT-NIGHT-{i:03d}",
        "uav_id_template": "UAV-B{slot}",
        "uav_slot_modulus": 3,
        "mission_type": "inspection",
        "asset_class": "MAINTENANCE_SAMPLE",
        "description": (
            "Night-time inspection flights. Low-visibility warnings are recorded "
            "as anomalies, so the governance engine assigns internal-only use: "
            "all flights are non-tradable despite full mission completion."
        ),
        "expected_tradable": False,
        "governance_path": "standard_governance",
        "injected_violations": [],
        "emergent_violations": [],
        "parameter_distributions": {
            "completed": {"dist": "constant", "value": True},
            "completion_pct": {"dist": "linear", "base": 95.0, "step": -1.0},
            "telemetry_points": {"dist": "linear", "base": 700, "step": 40},
            "wind_speed_mps": {"dist": "linear", "base": 2.0, "step": 0.5},
            "visibility_km": {"dist": "linear", "base": 4.0, "step": 0.3},
            "weather": {"dist": "constant", "value": "clear_night"},
            "duration_min": {"dist": "linear", "base": 18, "step": 1},
            "anomalies": {"dist": "constant", "value": ["low_visibility_warning"]},
            "risk_events": {"dist": "constant", "value": ["nighttime_operation"]},
            "no_fly_zone_incursions": {"dist": "constant", "value": 0},
        },
    },

    # ── 3. Weather disturbance ──────────────────────────────────────────────
    {
        "tag": "weather_disturbance",
        "count": 10,
        "flight_id_template": "FLT-WEATHER-{i:03d}",
        "uav_id_template": "UAV-C{slot}",
        "uav_slot_modulus": 3,
        "mission_type": "delivery",
        "asset_class": "WEATHER_OPERATION_SAMPLE",
        "description": (
            "Delivery flights under progressively degrading weather. Completion "
            "percentage drops with index. Turbulence/gust anomalies cause the "
            "governance engine to assign internal-only use: all flights are "
            "non-tradable."
        ),
        "expected_tradable": False,
        "governance_path": "standard_governance",
        "injected_violations": [],
        "emergent_violations": [],
        "parameter_distributions": {
            "completed": {"dist": "constant", "value": True},
            "completion_pct": {"dist": "linear", "base": 92.0, "step": -3.0},
            "telemetry_points": {"dist": "linear", "base": 500, "step": 30},
            "wind_speed_mps": {"dist": "linear", "base": 10.0, "step": 1.5},
            "visibility_km": {"dist": "linear", "base": 3.0, "step": -0.1},
            "weather": {"dist": "cycle", "values": ["rainy", "stormy", "foggy"]},
            "duration_min": {"dist": "linear", "base": 25, "step": 1},
            "anomalies": {"dist": "constant", "value": ["turbulence", "gust_warning"]},
            "risk_events": {"dist": "constant", "value": ["weather_degradation", "wind_shear"]},
            "no_fly_zone_incursions": {"dist": "constant", "value": 0},
        },
    },

    # ── 4. Near-NFZ ─────────────────────────────────────────────────────────
    {
        "tag": "near_nfz",
        "count": 8,
        "flight_id_template": "FLT-NFZ-{i:03d}",
        "uav_id_template": "UAV-D{slot}",
        "uav_slot_modulus": 2,
        "mission_type": "patrol",
        "asset_class": "COMPLIANCE_RECORD",
        "description": (
            "Patrol flights near no-fly zone boundaries. Flights 0-1 are clean; "
            "nfz_proximity anomalies start at i=2 (internal-only), and from i=3 "
            "onwards nfz_proximity_warning violations are injected and incursion "
            "count increases, blocking asset transfer."
        ),
        "expected_tradable": False,
        "governance_path": "mixed_pass_non_transfer",
        "injected_violations": [
            {
                "violation": "nfz_proximity_warning",
                "condition": "i >= 3",
                "rationale": "Simulates progressive encroachment; i<3 flights remain clean.",
            }
        ],
        "emergent_violations": [],
        "parameter_distributions": {
            "completed": {"dist": "constant", "value": True},
            "completion_pct": {"dist": "constant", "value": 100.0},
            "telemetry_points": {"dist": "linear", "base": 900, "step": 20},
            "wind_speed_mps": {"dist": "constant", "value": 5.0},
            "visibility_km": {"dist": "constant", "value": 8.0},
            "weather": {"dist": "constant", "value": "overcast"},
            "duration_min": {"dist": "linear", "base": 30, "step": 1},
            "anomalies": {"dist": "threshold", "value": ["nfz_proximity"], "threshold": 2},
            "risk_events": {"dist": "constant", "value": ["nfz_buffer_entry"]},
            "no_fly_zone_incursions": {"dist": "threshold_ramp", "threshold": 3},
        },
        "_notes": "First 2 flights (i<2) are tradable; anomalies from i=2 and "
                  "injected violations from i=3 make the remainder non-tradable.",
        "tradable_count": 2,
    },

    # ── 5. Anomaly-rich maintenance ─────────────────────────────────────────
    {
        "tag": "anomaly_maintenance",
        "count": 10,
        "flight_id_template": "FLT-MAINT-{i:03d}",
        "uav_id_template": "UAV-E{slot}",
        "uav_slot_modulus": 3,
        "mission_type": "maintenance_check",
        "asset_class": "MAINTENANCE_SAMPLE",
        "description": (
            "Maintenance-check flights with escalating hardware anomalies drawn "
            "from a pool of 5 sensor/actuator faults. No violations are injected; "
            "non-tradability emerges from accumulated anomaly scoring in the pipeline."
        ),
        "expected_tradable": False,
        "governance_path": "standard_governance",
        "injected_violations": [],
        "emergent_violations": [
            {
                "mechanism": "anomaly_accumulation",
                "pool": ["motor_vibration", "battery_temp_high", "gps_drift",
                         "compass_interference", "esc_warning"],
                "selection": "pool[: (i % 4) + 1]",
                "rationale": (
                    "Anomaly count grows with i; governance engine blocks promotion "
                    "once the weighted anomaly score exceeds the threshold."
                ),
            }
        ],
        "parameter_distributions": {
            "completed": {"dist": "constant", "value": True},
            "completion_pct": {"dist": "linear", "base": 88.0, "step": -1.0},
            "telemetry_points": {"dist": "linear", "base": 800, "step": 25},
            "wind_speed_mps": {"dist": "linear", "base": 3.0, "step": 0.2},
            "visibility_km": {"dist": "constant", "value": 12.0},
            "weather": {"dist": "constant", "value": "clear"},
            "duration_min": {"dist": "linear", "base": 12, "step": 1},
            "risk_events": {"dist": "constant", "value": ["equipment_warning"]},
            "no_fly_zone_incursions": {"dist": "constant", "value": 0},
        },
    },

    # ── 6. Emergency logistics ──────────────────────────────────────────────
    {
        "tag": "emergency_logistics",
        "count": 8,
        "flight_id_template": "FLT-EMER-{i:03d}",
        "uav_id_template": "UAV-F{slot}",
        "uav_slot_modulus": 2,
        "mission_type": "emergency_delivery",
        "asset_class": "RISK_DATASET",
        "description": (
            "Emergency delivery flights. Flights 0-4 complete nominally and remain "
            "tradable; from i=5 altitude_exceedance is injected and completion "
            "drops below threshold, blocking the remaining three flights."
        ),
        "expected_tradable": False,
        "governance_path": "mission_failure",
        "_notes": "Flights i<5 are tradable (5 total); i>=5 carry an injected "
                  "violation and are non-tradable (3 total).",
        "tradable_count": 5,
        "injected_violations": [
            {
                "violation": "altitude_exceedance",
                "condition": "i >= 5",
                "rationale": "Models abort scenario where UAV climbs above corridor ceiling.",
            }
        ],
        "emergent_violations": [],
        "parameter_distributions": {
            "completed": {"dist": "threshold_bool", "threshold": 5},
            "completion_pct": {"dist": "threshold_value",
                               "below_value": 100.0,
                               "above_expr": "40.0 + i * 5",
                               "threshold": 5},
            "telemetry_points": {"dist": "linear", "base": 400, "step": 30},
            "wind_speed_mps": {"dist": "linear", "base": 5.0, "step": 1.0},
            "visibility_km": {"dist": "constant", "value": 10.0},
            "weather": {"dist": "constant", "value": "clear"},
            "duration_min": {"dist": "linear", "base": 8, "step": 1},
            "anomalies": {"dist": "threshold", "value": ["mission_abort"], "threshold": 5},
            "risk_events": {"dist": "threshold",
                            "value": ["priority_corridor", "emergency_landing"],
                            "below_value": ["priority_corridor"],
                            "threshold": 6},
            "no_fly_zone_incursions": {"dist": "constant", "value": 0},
        },
    },

    # ── 7. Low-quality / incomplete ─────────────────────────────────────────
    {
        "tag": "low_quality",
        "count": 12,
        "flight_id_template": "FLT-LOWQ-{i:03d}",
        "uav_id_template": "UAV-G{slot}",
        "uav_slot_modulus": 2,
        "mission_type": "survey",
        "asset_class": "FLIGHT_EVIDENCE",
        "description": (
            "Incomplete survey flights with injected data-quality violations. "
            "data_gap is always present; sensor_failure is added for odd-indexed "
            "flights. Completion rises linearly from 20% but never reaches threshold."
        ),
        "expected_tradable": False,
        "governance_path": "quality_failure",
        "injected_violations": [
            {
                "violation": "data_gap",
                "condition": "always",
                "rationale": "Models partial telemetry dropout common in low-resource missions.",
            },
            {
                "violation": "sensor_failure",
                "condition": "i % 2 == 1",
                "rationale": "Alternating sensor failure exercises the multi-violation path.",
            },
        ],
        "emergent_violations": [],
        "parameter_distributions": {
            "completed": {"dist": "constant", "value": False},
            "completion_pct": {"dist": "linear", "base": 20.0, "step": 5.0},
            "telemetry_points": {"dist": "linear", "base": 50, "step": 20},
            "wind_speed_mps": {"dist": "linear", "base": 7.0, "step": 0.5},
            "visibility_km": {"dist": "linear", "base": 1.0, "step": 0.1},
            "weather": {"dist": "constant", "value": "foggy"},
            "duration_min": {"dist": "linear", "base": 3, "step": 1},
            "anomalies": {"dist": "constant", "value": ["telemetry_loss", "gps_failure"]},
            "risk_events": {"dist": "constant", "value": []},
            "no_fly_zone_incursions": {"dist": "constant", "value": 0},
        },
    },

    # ── 8. Rights-conflict aggregation ──────────────────────────────────────
    {
        "tag": "rights_conflict",
        "count": 8,
        "flight_id_template": "FLT-RIGHTS-{i:03d}",
        "uav_id_template": "UAV-H{slot}",
        "uav_slot_modulus": 3,
        "mission_type": "commercial_survey",
        "asset_class": "AUDIT_READY_PACKAGE",
        "description": (
            "High-quality commercial survey flights that pass all checks and "
            "remain tradable. Each flight's rights profile requires "
            "desensitization before transfer, so the GOV-002 rights rule flags "
            "every asset with a non-blocking desensitization obligation, "
            "materialized as a GovernanceDecision."
        ),
        "expected_tradable": True,
        "governance_path": "rights_obligation",
        "injected_violations": [],
        "emergent_violations": [
            {
                "mechanism": "gov002_desensitization_obligation",
                "rationale": (
                    "GOV-002 fires on every tradable asset whose rights profile "
                    "requires desensitization, flagging a pre-transfer obligation "
                    "without blocking promotion."
                ),
            }
        ],
        "parameter_distributions": {
            "completed": {"dist": "constant", "value": True},
            "completion_pct": {"dist": "linear", "base": 97.0, "step": 0.3},
            "telemetry_points": {"dist": "linear", "base": 1400, "step": 50},
            "wind_speed_mps": {"dist": "constant", "value": 3.0},
            "visibility_km": {"dist": "constant", "value": 15.0},
            "weather": {"dist": "constant", "value": "clear"},
            "duration_min": {"dist": "linear", "base": 35, "step": 3},
            "anomalies": {"dist": "constant", "value": []},
            "risk_events": {"dist": "constant", "value": []},
            "no_fly_zone_incursions": {"dist": "constant", "value": 0},
        },
    },

    # ── 9. Beyond-VLOS ──────────────────────────────────────────────────────
    {
        "tag": "beyond_vlos",
        "count": 15,
        "flight_id_template": "FLT-BVLOS-{i:03d}",
        "uav_id_template": "UAV-J{slot}",
        "uav_slot_modulus": 4,
        "mission_type": "long_range_survey",
        "asset_class": "ROUTE_OPTIMIZATION_SAMPLE",
        "description": (
            "Long-range BVLOS survey flights. Flights 0-9 complete with relay "
            "hand-offs and remain tradable; link_degradation anomalies from i=10 "
            "demote flights 10-11 to internal-only; from i=12, range_exceedance "
            "is injected and completion drops, blocking promotion."
        ),
        "expected_tradable": True,
        "governance_path": "range_link_edge_case",
        "injected_violations": [
            {
                "violation": "range_exceedance",
                "condition": "i >= 12",
                "rationale": "Models three flights that push beyond the licensed BVLOS corridor.",
            }
        ],
        "emergent_violations": [],
        "parameter_distributions": {
            "completed": {"dist": "threshold_bool", "threshold": 12},
            "completion_pct": {"dist": "threshold_value",
                               "below_value": 100.0,
                               "above_expr": "70.0",
                               "threshold": 12},
            "telemetry_points": {"dist": "linear", "base": 2000, "step": 100},
            "wind_speed_mps": {"dist": "linear", "base": 4.0, "step": 0.4},
            "visibility_km": {"dist": "linear", "base": 10.0, "step": -0.3},
            "weather": {"dist": "cycle", "values": ["clear", "overcast", "hazy"]},
            "duration_min": {"dist": "linear", "base": 45, "step": 5},
            "anomalies": {"dist": "threshold", "value": ["link_degradation"], "threshold": 10},
            "risk_events": {"dist": "constant", "value": ["beyond_vlos", "relay_handoff"]},
            "no_fly_zone_incursions": {"dist": "constant", "value": 0},
        },
        "_notes": "Flights i<10 are tradable (10 total); i=10,11 are demoted by "
                  "link_degradation anomalies; i>=12 carry an injected violation.",
        "tradable_count": 10,
    },

    # ── 10. Urban corridor multi-stop ────────────────────────────────────────
    {
        "tag": "urban_corridor",
        "count": 14,
        "flight_id_template": "FLT-URBAN-{i:03d}",
        "uav_id_template": "UAV-K{slot}",
        "uav_slot_modulus": 5,
        "mission_type": "urban_delivery",
        "asset_class_expr": (
            "ROUTE_OPTIMIZATION_SAMPLE if i < 10 else RISK_DATASET"
        ),
        "description": (
            "Urban multi-stop delivery flights in a dense corridor. Flights 0-7 "
            "remain tradable; obstacle_proximity anomalies from i=8 demote "
            "flights to internal-only; altitude_exceedance is injected from "
            "i=11; NFZ incursions start at i=12."
        ),
        "expected_tradable": False,
        "governance_path": "urban_density_nfz",
        "_notes": "Flights i<8 are tradable (8 total); i=8..10 are demoted by "
                  "anomalies; i>=11 carry an injected violation.",
        "tradable_count": 8,
        "injected_violations": [
            {
                "violation": "altitude_exceedance",
                "condition": "i >= 11",
                "rationale": "Urban canyon effect forces UAV above certified ceiling.",
            }
        ],
        "emergent_violations": [
            {
                "mechanism": "obstacle_proximity_accumulation",
                "condition": "i >= 8",
                "rationale": "Dense building layout causes proximity warnings above i=8.",
            }
        ],
        "parameter_distributions": {
            "completed": {"dist": "threshold_bool", "threshold": 11},
            "completion_pct": {"dist": "threshold_value",
                               "below_value": 100.0,
                               "above_expr": "55.0",
                               "threshold": 11},
            "telemetry_points": {"dist": "linear", "base": 600, "step": 50},
            "wind_speed_mps": {"dist": "linear", "base": 3.0, "step": 0.3},
            "visibility_km": {"dist": "linear", "base": 12.0, "step": -0.4},
            "weather": {"dist": "threshold_value",
                        "below_value": "clear",
                        "above_expr": "overcast",
                        "threshold": 7},
            "duration_min": {"dist": "linear", "base": 15, "step": 2},
            "anomalies": {"dist": "threshold", "value": ["obstacle_proximity"], "threshold": 8},
            "risk_events": {"dist": "threshold",
                            "value": ["urban_density", "obstacle_avoidance"],
                            "below_value": ["urban_density"],
                            "threshold": 6},
            "no_fly_zone_incursions": {"dist": "threshold_bool_int", "threshold": 12},
        },
    },
]

# ---------------------------------------------------------------------------
# Convenience index
# ---------------------------------------------------------------------------

SPEC_BY_TAG: dict[str, dict] = {s["tag"]: s for s in SCENARIO_SPECS}

TOTAL_FLIGHTS: int = sum(s["count"] for s in SCENARIO_SPECS)
