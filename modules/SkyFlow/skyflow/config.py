"""Centralized configuration dataclass matching paper hyperparameters."""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ModelConfig:
    num_layers: int = 4
    embed_dim: int = 128
    num_heads: int = 4
    temporal_dim: int = 32
    recurrent_dim: int = 64
    dropout: float = 0.1
    num_relation_types: int = 6


@dataclass
class DataConfig:
    num_uavs: int = 500
    num_sectors: int = 64
    num_weather_cells: int = 36
    num_restricted_zones: int = 12
    uav_feature_dim: int = 23
    sector_feature_dim: int = 8
    weather_feature_dim: int = 12
    observation_window: int = 10
    lookahead_seconds: float = 30.0
    conflict_h_sep_m: float = 10.0
    conflict_v_sep_m: float = 3.0
    sim_freq_hz: float = 10.0
    grid_size_m: float = 5000.0
    scenario_minutes_train: int = 3360
    scenario_minutes_val: int = 720
    scenario_minutes_test: int = 720


@dataclass
class TrainingConfig:
    epochs: int = 150
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    focal_gamma: float = 2.0
    conflict_threshold: float = 0.42
    warmup_steps: int = 1000
    seed: int = 42
    num_seeds: int = 5
    device: str = "auto"


@dataclass
class SkyFlowConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: str = "outputs"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SkyFlowConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        cfg = cls()
        for section_name in ("model", "data", "training"):
            if section_name in raw:
                section = getattr(cfg, section_name)
                for k, v in raw[section_name].items():
                    if hasattr(section, k):
                        setattr(section, k, v)
        if "output_dir" in raw:
            cfg.output_dir = raw["output_dir"]
        return cfg

    def to_yaml(self, path: str | Path) -> None:
        from dataclasses import asdict
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
