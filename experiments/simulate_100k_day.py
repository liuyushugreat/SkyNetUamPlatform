"""simulate_100k_day.py

Daily 100k-mission simulation for SkyNetUAM.

This script generates a reproducible synthetic workload for low-altitude UAM/drone
operations with mission lifecycle outcomes (approved/delayed/completed/failed/denied)
under corridor-capacity and regulatory-permission constraints.

Outputs (written under experiments/outputs/):
- flights_day.csv: per-mission dataset (100,000 rows)
- events_day.csv: lifecycle events (Created/Scheduled/Active/Completed/Failed)
- kpi_summary.csv: summary KPI table
- fig_volume.png: hourly demand vs approvals
- fig_corridor_heatmap.png: corridor-time utilization heatmap
- fig_latency_cdf.png: scheduling latency + settlement finality CDF
- fig_outcomes.png: mission outcomes distribution

Run:
  python experiments/simulate_100k_day.py

Dependencies:
  numpy, pandas, matplotlib, seaborn

"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


@dataclass
class SimConfig:
    n_missions: int = 100_000
    seed: int = 42

    # City / corridor model
    n_corridors: int = 24
    bin_minutes: int = 5
    corridor_capacity_per_bin: int = 35  # induces peak congestion
    max_delay_minutes: int = 20

    # Timing
    day_minutes: int = 24 * 60

    # Vehicle / mission
    cruise_kmh_mean: float = 55.0
    cruise_kmh_sd: float = 12.0

    # Reliability
    base_fail_prob: float = 0.004
    risk_fail_weight: float = 0.020
    wind_fail_weight: float = 0.010

    # Compliance
    base_violation_prob: float = 0.002
    wind_violation_weight: float = 0.006

    # Regulatory pre-approval (permission outcome)
    base_reg_denial_prob: float = 0.010
    wind_reg_denial_weight: float = 0.030
    risk_reg_denial_weight: float = 0.020

    # Optional persistence mode (simulated finality for paper-style comparison)
    persistence_mode: str = "l2"  # off | devnet | l2 | l1


def _set_style():
    sns.set_context("paper")
    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.titlesize"] = 12


def _ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def _sample_minutes_of_day(n: int, rng: np.random.Generator) -> np.ndarray:
    """Mixture model to create realistic diurnal peaks."""
    weights = np.array([0.42, 0.38, 0.20])
    comp = rng.choice(3, size=n, p=weights)

    minutes = np.empty(n, dtype=float)
    minutes[comp == 0] = rng.normal(loc=9 * 60, scale=90, size=(comp == 0).sum())
    minutes[comp == 1] = rng.normal(loc=18 * 60, scale=110, size=(comp == 1).sum())
    minutes[comp == 2] = rng.uniform(0, 24 * 60, size=(comp == 2).sum())

    return np.clip(minutes, 0, 24 * 60 - 1)


def _weather_by_hour(rng: np.random.Generator) -> pd.DataFrame:
    hours = np.arange(24)
    base = 0.35 + 0.25 * np.sin((hours - 6) / 24 * 2 * np.pi)
    wind = np.clip(base + rng.normal(0, 0.08, size=24), 0, 1)
    return pd.DataFrame({"hour": hours, "wind_index": wind})


def _persistence_finality_seconds(mode: str) -> float:
    if mode == "off":
        return 0.02
    if mode == "devnet":
        return 0.08
    if mode == "l2":
        return 2.3
    if mode == "l1":
        return 13.5
    raise ValueError(f"Unknown persistence_mode: {mode}")


def simulate(cfg: SimConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)

    request_minute = _sample_minutes_of_day(cfg.n_missions, rng)
    request_minute.sort()

    o_x, o_y = rng.uniform(0, 1, size=cfg.n_missions), rng.uniform(0, 1, size=cfg.n_missions)
    d_x, d_y = rng.uniform(0, 1, size=cfg.n_missions), rng.uniform(0, 1, size=cfg.n_missions)

    raw_w = np.linspace(2.0, 0.6, cfg.n_corridors)
    w = raw_w / raw_w.sum()
    w = w[rng.permutation(cfg.n_corridors)]
    corridor = rng.choice(cfg.n_corridors, size=cfg.n_missions, p=w)

    dist_km = np.sqrt((o_x - d_x) ** 2 + (o_y - d_y) ** 2) * 18
    dist_km = np.clip(dist_km + rng.lognormal(mean=-0.6, sigma=0.5, size=cfg.n_missions) * 2.2, 0.8, 35.0)

    speed_kmh = np.clip(rng.normal(cfg.cruise_kmh_mean, cfg.cruise_kmh_sd, size=cfg.n_missions), 25, 90)
    duration_min = np.clip((dist_km / speed_kmh) * 60, 3.0, 55.0)

    base_risk = np.clip(rng.beta(2.2, 6.5, size=cfg.n_missions), 0, 1)

    weather = _weather_by_hour(rng)
    request_hour = (request_minute // 60).astype(int)
    wind_index = weather.loc[request_hour, "wind_index"].to_numpy()

    risk = np.clip(base_risk + 0.35 * wind_index, 0, 1)

    reg_denial_p = cfg.base_reg_denial_prob + cfg.wind_reg_denial_weight * wind_index + cfg.risk_reg_denial_weight * risk
    reg_denial_p = np.clip(reg_denial_p, 0, 0.20)
    reg_denied = rng.random(cfg.n_missions) < reg_denial_p

    n_bins = cfg.day_minutes // cfg.bin_minutes
    bin_index = np.ceil(request_minute / cfg.bin_minutes).astype(int)

    cap = np.zeros((cfg.n_corridors, n_bins), dtype=int)
    scheduled_bin = np.full(cfg.n_missions, -1, dtype=int)
    denied = reg_denied.copy()
    denial_reason = np.where(reg_denied, "REGULATORY", "")

    max_shift_bins = int(cfg.max_delay_minutes // cfg.bin_minutes)

    for i in range(cfg.n_missions):
        if denied[i]:
            continue
        c = corridor[i]
        b0 = int(bin_index[i])

        placed = False
        for shift in range(max_shift_bins + 1):
            b = b0 + shift
            if b >= n_bins:
                break
            if cap[c, b] < cfg.corridor_capacity_per_bin:
                cap[c, b] += 1
                scheduled_bin[i] = b
                placed = True
                break

        if not placed:
            denied[i] = True
            denial_reason[i] = "CAPACITY"

    scheduled_minute = (scheduled_bin * cfg.bin_minutes).astype(float)
    scheduled_minute[denied] = np.nan

    delayed = (~denied) & (scheduled_bin > bin_index)

    takeoff_minute = scheduled_minute + np.clip(rng.normal(0.6, 0.35, size=cfg.n_missions), 0, 4)
    takeoff_minute[denied] = np.nan

    end_minute = takeoff_minute + duration_min

    fail_prob = cfg.base_fail_prob + cfg.risk_fail_weight * risk + cfg.wind_fail_weight * wind_index
    fail_prob = np.clip(fail_prob, 0, 0.25)
    failed = (~denied) & (rng.random(cfg.n_missions) < fail_prob)

    completed = (~denied) & (~failed)

    violation_prob = cfg.base_violation_prob + cfg.wind_violation_weight * wind_index
    violation_prob = np.clip(violation_prob, 0, 0.20)
    violated = (~denied) & (rng.random(cfg.n_missions) < violation_prob)

    finality_s = _persistence_finality_seconds(cfg.persistence_mode)
    settlement_sec = np.where(
        ~denied,
        np.clip(rng.normal(finality_s, finality_s * 0.15 + 0.02, size=cfg.n_missions), 0.02, None),
        np.nan,
    )

    value_score = np.clip(60 + 12 * dist_km + 80 * (1 - risk) + rng.normal(0, 15, size=cfg.n_missions), 10, None)

    flights = pd.DataFrame(
        {
            "mission_id": [f"m-{i:06d}" for i in range(cfg.n_missions)],
            "request_minute": request_minute,
            "scheduled_minute": scheduled_minute,
            "takeoff_minute": takeoff_minute,
            "end_minute": end_minute,
            "corridor_id": corridor,
            "dist_km": dist_km,
            "duration_min": duration_min,
            "risk_score": risk,
            "wind_index": wind_index,
            "approved": ~denied,
            "delayed": delayed,
            "denial_reason": np.where(~denied, "", denial_reason),
            "failed": failed,
            "completed": completed,
            "violation": violated,
            "settlement_finality_s": settlement_sec,
            "value_score": value_score,
        }
    )

    events_rows = []
    for _, row in flights.iterrows():
        mid = row["mission_id"]
        t_req = float(row["request_minute"])
        events_rows.append((mid, "Created", t_req))

        if not row["approved"]:
            continue

        events_rows.append((mid, "Scheduled", float(row["scheduled_minute"])))
        events_rows.append((mid, "Active", float(row["takeoff_minute"])))
        if row["failed"]:
            events_rows.append((mid, "Failed", float(row["end_minute"])))
        else:
            events_rows.append((mid, "Completed", float(row["end_minute"])))

    events = pd.DataFrame(events_rows, columns=["mission_id", "state", "minute"])

    approved_rate = flights["approved"].mean()
    delay_rate = flights.loc[flights["approved"], "delayed"].mean()
    fail_rate = flights.loc[flights["approved"], "failed"].mean()
    violation_rate = flights.loc[flights["approved"], "violation"].mean()

    sched_latency_min = (flights["scheduled_minute"] - flights["request_minute"]).dropna()

    kpi = pd.DataFrame(
        {
            "metric": [
                "Missions/day",
                "Approval rate",
                "Delay rate (approved)",
                "Failure rate (approved)",
                "Violation rate (approved)",
                "Median scheduling latency (min)",
                "P95 scheduling latency (min)",
                "Median settlement finality (s)",
            ],
            "value": [
                cfg.n_missions,
                approved_rate,
                delay_rate,
                fail_rate,
                violation_rate,
                float(np.median(sched_latency_min)),
                float(np.quantile(sched_latency_min, 0.95)),
                float(np.nanmedian(flights["settlement_finality_s"])),
            ],
        }
    )

    return flights, events, kpi


def make_figures(flights: pd.DataFrame, cfg: SimConfig):
    _set_style()

    flights = flights.copy()
    flights["request_hour"] = (flights["request_minute"] // 60).astype(int)

    hourly = (
        flights.groupby("request_hour")
        .agg(requests=("mission_id", "count"), approved=("approved", "sum"))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(hourly["request_hour"], hourly["requests"], label="Requests", linewidth=2.2, color="#2E86AB")
    ax.plot(hourly["request_hour"], hourly["approved"], label="Approved", linewidth=2.2, color="#06A77D")
    ax.fill_between(hourly["request_hour"], hourly["approved"], hourly["requests"], alpha=0.12, color="#2E86AB")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Missions")
    ax.set_title("Daily Workload (100k missions/day): Demand vs Approval")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_volume.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    bin_minutes = cfg.bin_minutes
    flights_ok = flights.loc[flights["approved"]].copy()
    flights_ok["bin"] = (flights_ok["scheduled_minute"] // bin_minutes).astype(int)

    util = flights_ok.pivot_table(index="corridor_id", columns="bin", values="mission_id", aggfunc="count", fill_value=0)
    util_ratio = util / float(cfg.corridor_capacity_per_bin)

    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    sns.heatmap(util_ratio, cmap="viridis", ax=ax, cbar_kws={"label": "Utilization (fraction of capacity)"})
    ax.set_xlabel(f"Time bins ({bin_minutes}-min)")
    ax.set_ylabel("Corridor ID")
    ax.set_title("Air-Corridor Utilization Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_corridor_heatmap.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    sched_latency = (flights["scheduled_minute"] - flights["request_minute"]).dropna().to_numpy()
    settlement = flights.loc[flights["approved"], "settlement_finality_s"].dropna().to_numpy()

    def cdf(x: np.ndarray):
        xs = np.sort(x)
        ys = np.linspace(0, 1, len(xs), endpoint=True)
        return xs, ys

    x1, y1 = cdf(sched_latency)
    x2, y2 = cdf(settlement)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(x1, y1, label="Scheduling latency (min)", linewidth=2.2, color="#A23B72")
    ax2 = ax.twiny()
    ax2.plot(x2, y2, label=f"Settlement finality (s) [{cfg.persistence_mode.upper()}]", linewidth=2.2, color="#F18F01")

    ax.set_xlabel("Scheduling latency (minutes)")
    ax.set_ylabel("CDF")
    ax2.set_xlabel("Settlement finality (seconds)")

    lines = ax.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="lower right", frameon=True)

    ax.set_title("Latency Distributions (CDF)")
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_latency_cdf.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    outcomes = {
        "Completed": int(flights["completed"].sum()),
        "Failed": int(flights["failed"].sum()),
        "Denied": int((~flights["approved"]).sum()),
    }

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    labels = list(outcomes.keys())
    vals = np.array(list(outcomes.values()), dtype=float)
    colors = ["#06A77D", "#C73E1D", "#6B7280"]
    ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Missions")
    ax.set_title("Mission Outcomes (Daily)")
    for i, v in enumerate(vals):
        ax.text(i, v + vals.max() * 0.015, f"{int(v):,}", ha="center", va="bottom", fontweight="bold", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_outcomes.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def main():
    _ensure_out_dir()
    cfg = SimConfig()
    flights, events, kpi = simulate(cfg)

    flights.to_csv(os.path.join(OUT_DIR, "flights_day.csv"), index=False)
    events.to_csv(os.path.join(OUT_DIR, "events_day.csv"), index=False)
    kpi.to_csv(os.path.join(OUT_DIR, "kpi_summary.csv"), index=False)

    make_figures(flights, cfg)

    print("Simulation complete")
    print(f"  - {os.path.join(OUT_DIR, 'flights_day.csv')}")


if __name__ == "__main__":
    main()
