"""Typed configuration objects for SkyCert experiments.

The YAML files under ``configs/`` are loaded into the dataclasses below.
Keeping configuration typed (rather than raw dicts) makes it easy to
validate that the values used to produce a given audit artifact match
the ones described in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    num_train: int = 4000
    num_calib: int = 1500
    num_test: int = 3000
    num_features: int = 16
    num_classes: int = 4
    class_prior: list[float] = field(
        default_factory=lambda: [0.55, 0.25, 0.15, 0.05]
    )


@dataclass
class BaseModelConfig:
    type: str = "logistic"
    hidden: int = 64
    l2: float = 5e-4
    max_iter: int = 500


@dataclass
class SymbolicRule:
    feature: int
    op: str
    thr: float
    delta: list[float]


@dataclass
class SymbolicConfig:
    lambda_: float = 0.35
    rules: list[SymbolicRule] = field(default_factory=list)


@dataclass
class ConformalConfig:
    alpha: float = 0.1
    score: str = "aps"  # {aps, lac}


@dataclass
class MartingaleConfig:
    type: str = "simple_jumper"
    epsilon: float = 0.92
    threshold: float = 20.0


@dataclass
class PolicyConfig:
    max_set_fraction: float = 0.75
    escalate_on_martingale: bool = True


@dataclass
class AssuranceConfig:
    conformal: ConformalConfig = field(default_factory=ConformalConfig)
    martingale: MartingaleConfig = field(default_factory=MartingaleConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)


@dataclass
class ThreatConfig:
    name: str
    kind: str  # {none, rule_flip, rule_inject, feature_attack, covariate_shift}
    strength: float = 0.0


@dataclass
class SkyCertConfig:
    seed: int = 20260417
    data: DataConfig = field(default_factory=DataConfig)
    base_model: BaseModelConfig = field(default_factory=BaseModelConfig)
    symbolic: SymbolicConfig = field(default_factory=SymbolicConfig)
    assurance: AssuranceConfig = field(default_factory=AssuranceConfig)
    threats: list[ThreatConfig] = field(default_factory=list)
    output_dir: str = "outputs"

    @classmethod
    def load(cls, path: str | Path) -> "SkyCertConfig":
        with open(path, "r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SkyCertConfig":
        data = DataConfig(**raw.get("data", {}))
        base_model = BaseModelConfig(**raw.get("base_model", {}))
        sym_raw = dict(raw.get("symbolic", {}))
        # YAML uses ``lambda`` which is a Python reserved word.
        if "lambda" in sym_raw:
            sym_raw["lambda_"] = sym_raw.pop("lambda")
        rule_dicts = sym_raw.pop("rules", [])
        symbolic = SymbolicConfig(
            rules=[SymbolicRule(**r) for r in rule_dicts], **sym_raw
        )
        assurance_raw = raw.get("assurance", {})
        assurance = AssuranceConfig(
            conformal=ConformalConfig(**assurance_raw.get("conformal", {})),
            martingale=MartingaleConfig(**assurance_raw.get("martingale", {})),
            policy=PolicyConfig(**assurance_raw.get("policy", {})),
        )
        threats = [ThreatConfig(**t) for t in raw.get("threats", [])]
        return cls(
            seed=int(raw.get("seed", 20260417)),
            data=data,
            base_model=base_model,
            symbolic=symbolic,
            assurance=assurance,
            threats=threats,
            output_dir=str(raw.get("output_dir", "outputs")),
        )
