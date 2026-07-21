"""Baseline comparison for SkyCert.

Implements two standard uncertainty-based abstention baselines (MSP
threshold and entropy threshold) and compares them against the full
SkyCert pipeline under the distribution_shift scenario, using matched
abstention rates for fair evaluation.

Also compares three *backbones* under the same threat -- the full
neuro-symbolic reasoner, a purely neural scorer (symbolic weight
lambda = 0), and a purely symbolic rule engine (neural logits zeroed)
-- each wrapped by the identical SkyCert assurance layer, so the
contribution of the hybrid backbone can be separated from the
contribution of the assurance layer.

Also produces Pareto sweep data (abstain rate vs critical-miss rate) by
varying each method's operating-point knob, used for Figure 5 in the
paper.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from skycert.assurance.policy import DecisionKind
from skycert.base import (
    NeuralRiskScorer,
    NeuroSymbolicRiskReasoner,
    SymbolicRuleEngine,
)
from skycert.config import SkyCertConfig
from skycert.data import apply_threat, make_uam_dataset
from skycert.metrics import (
    abstain_error_rate,
    average_set_size,
    critical_error_rate,
    empirical_coverage,
)
from skycert.pipeline import SkyCertPipeline
from skycert.utils import ensure_dir, safe_json


def _build_stream(
    config: SkyCertConfig, X_test: np.ndarray
) -> tuple[np.ndarray, int]:
    """Build the distribution_shift stream identically to run_experiment.py."""
    threat_seed = int.from_bytes(
        hashlib.sha256(b"distribution_shift").digest()[:4], "little"
    )
    rng = np.random.default_rng(config.seed + threat_seed)
    X_stream = X_test.copy()
    half = X_stream.shape[0] // 2
    updates = apply_threat("covariate_shift", X=X_stream[half:], strength=0.8, rng=rng)
    X_stream[half:] = updates["X"]
    return X_stream, half


def _entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


class _ZeroScorer:
    """Stand-in neural scorer emitting all-zero logits.

    Used to build a *purely symbolic* backbone: the reasoner's output then
    reduces to softmax(lambda * symbolic_logits).
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes

    def logits(self, X: np.ndarray) -> np.ndarray:
        return np.zeros((X.shape[0], self.num_classes), dtype=np.float64)


def _backbone_metrics(
    name: str,
    reasoner: NeuroSymbolicRiskReasoner,
    config: SkyCertConfig,
    data,
    X_stream: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Wrap a backbone with the identical SkyCert layer and measure it."""
    pipe = SkyCertPipeline(reasoner=reasoner, config=config, audit_path=None)
    pipe.calibrate(data.X_calib, data.y_calib)
    decisions = pipe.run_batch(X_stream)
    pipe.close()
    probs = pipe.predict_proba(X_stream)
    preds = probs.argmax(axis=1)
    set_mask = pipe.predict_sets(X_stream)
    abstained = np.array(
        [d.kind in {DecisionKind.ABSTAIN, DecisionKind.ALERT, DecisionKind.ESCALATE}
         for d in decisions]
    )
    return {
        "backbone": name,
        "top1_accuracy": float((preds == y_test).mean()),
        "coverage": empirical_coverage(set_mask, y_test),
        "avg_set_size": average_set_size(set_mask),
        "critical_error_rate_base": critical_error_rate(preds, y_test),
        "critical_error_rate_after_abstain": abstain_error_rate(
            preds, y_test, abstained
        ),
        "abstain_rate": float(abstained.mean()),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run baseline comparison")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-name", default="baselines.json")
    args = parser.parse_args(argv)

    config = SkyCertConfig.load(args.config)
    if args.seed is not None:
        config.seed = int(args.seed)
    out_dir = ensure_dir(config.output_dir)

    data = make_uam_dataset(
        num_train=config.data.num_train,
        num_calib=config.data.num_calib,
        num_test=config.data.num_test,
        num_features=config.data.num_features,
        num_classes=config.data.num_classes,
        class_prior=config.data.class_prior,
        seed=config.seed,
    )

    neural = NeuralRiskScorer(
        num_classes=data.num_classes,
        l2=config.base_model.l2,
        max_iter=config.base_model.max_iter,
        model_type=config.base_model.type,
        hidden=config.base_model.hidden,
    ).fit(data.X_train, data.y_train)
    symbolic = SymbolicRuleEngine(
        rules=config.symbolic.rules, num_classes=data.num_classes
    )
    reasoner = NeuroSymbolicRiskReasoner(
        neural=neural, symbolic=symbolic, lambda_=config.symbolic.lambda_
    )

    X_stream, change_point = _build_stream(config, data.X_test)
    y_test = data.y_test
    probs = reasoner.predict_proba(X_stream)
    preds = probs.argmax(axis=1)

    # --- Full SkyCert (reference) ---
    audit_dir = ensure_dir(out_dir / "audit_baselines")
    pipeline = SkyCertPipeline(
        reasoner=reasoner, config=config, audit_path=audit_dir / "full.jsonl"
    )
    pipeline.calibrate(data.X_calib, data.y_calib)
    decisions = pipeline.run_batch(X_stream)
    pipeline.close()

    skycert_abstained = np.array(
        [d.kind in {DecisionKind.ABSTAIN, DecisionKind.ALERT, DecisionKind.ESCALATE}
         for d in decisions]
    )
    skycert_set_mask = pipeline.predict_sets(X_stream)
    target_abstain = float(skycert_abstained.mean())

    # --- 1. MSP threshold (matched abstain rate) ---
    max_probs = probs.max(axis=1)
    msp_thr = float(np.quantile(max_probs, target_abstain))
    msp_abstained = max_probs <= msp_thr
    # fine-tune to get closest abstain rate
    if abs(msp_abstained.mean() - target_abstain) > 0.01:
        for t in np.linspace(max_probs.min(), max_probs.max(), 2000):
            cand = max_probs <= t
            if abs(cand.mean() - target_abstain) < abs(msp_abstained.mean() - target_abstain):
                msp_abstained = cand
                msp_thr = float(t)

    # --- 2. Entropy threshold (matched abstain rate) ---
    ent = _entropy(probs)
    ent_thr = float(np.quantile(ent, 1.0 - target_abstain))
    ent_abstained = ent >= ent_thr
    if abs(ent_abstained.mean() - target_abstain) > 0.01:
        for t in np.linspace(ent.min(), ent.max(), 2000):
            cand = ent >= t
            if abs(cand.mean() - target_abstain) < abs(ent_abstained.mean() - target_abstain):
                ent_abstained = cand
                ent_thr = float(t)

    # --- 3. Conformal-only (no martingale) ---
    cfg_conf = copy.deepcopy(config)
    cfg_conf.assurance.martingale.threshold = float("inf")
    pipe_conf = SkyCertPipeline(
        reasoner=reasoner, config=cfg_conf,
        audit_path=audit_dir / "conformal_only.jsonl",
    )
    pipe_conf.calibrate(data.X_calib, data.y_calib)
    dec_conf = pipe_conf.run_batch(X_stream)
    pipe_conf.close()
    conf_abstained = np.array(
        [d.kind in {DecisionKind.ABSTAIN, DecisionKind.ALERT, DecisionKind.ESCALATE}
         for d in dec_conf]
    )
    conf_set_mask = pipe_conf.predict_sets(X_stream)

    baselines = [
        {
            "method": "MSP threshold",
            "coverage": None,
            "avg_set_size": 1.0,
            "critical_error_rate_base": critical_error_rate(preds, y_test),
            "critical_error_rate_after_abstain": abstain_error_rate(preds, y_test, msp_abstained),
            "abstain_rate": float(msp_abstained.mean()),
        },
        {
            "method": "Entropy threshold",
            "coverage": None,
            "avg_set_size": 1.0,
            "critical_error_rate_base": critical_error_rate(preds, y_test),
            "critical_error_rate_after_abstain": abstain_error_rate(preds, y_test, ent_abstained),
            "abstain_rate": float(ent_abstained.mean()),
        },
        {
            "method": "Conformal-only",
            "coverage": empirical_coverage(conf_set_mask, y_test),
            "avg_set_size": average_set_size(conf_set_mask),
            "critical_error_rate_base": critical_error_rate(preds, y_test),
            "critical_error_rate_after_abstain": abstain_error_rate(preds, y_test, conf_abstained),
            "abstain_rate": float(conf_abstained.mean()),
        },
        {
            "method": "full SkyCert",
            "coverage": empirical_coverage(skycert_set_mask, y_test),
            "avg_set_size": average_set_size(skycert_set_mask),
            "critical_error_rate_base": critical_error_rate(preds, y_test),
            "critical_error_rate_after_abstain": abstain_error_rate(preds, y_test, skycert_abstained),
            "abstain_rate": float(skycert_abstained.mean()),
        },
    ]

    # --- Backbone comparison: neuro-symbolic vs pure-neural vs pure-symbolic ---
    # All three backbones are wrapped by the identical SkyCert layer so the
    # comparison isolates the contribution of the hybrid architecture.
    pure_neural = NeuroSymbolicRiskReasoner(
        neural=neural, symbolic=symbolic, lambda_=0.0
    )
    pure_symbolic = NeuroSymbolicRiskReasoner(
        neural=_ZeroScorer(data.num_classes), symbolic=symbolic, lambda_=1.0
    )
    backbones = [
        _backbone_metrics("neuro-symbolic", reasoner, config, data, X_stream, y_test),
        _backbone_metrics("pure neural", pure_neural, config, data, X_stream, y_test),
        _backbone_metrics("pure symbolic", pure_symbolic, config, data, X_stream, y_test),
    ]

    # --- Pareto sweep ---
    pareto: dict[str, list[dict[str, Any]]] = {"msp": [], "entropy": [], "skycert": []}

    for t in np.linspace(float(max_probs.min()), float(max_probs.max()), 80):
        ab = max_probs <= t
        ar = float(ab.mean())
        if ar < 0.01 or ar > 0.99:
            continue
        pareto["msp"].append({
            "abstain_rate": ar,
            "critical_error_after_abstain": abstain_error_rate(preds, y_test, ab),
        })

    for t in np.linspace(float(ent.min()), float(ent.max()), 80):
        ab = ent >= t
        ar = float(ab.mean())
        if ar < 0.01 or ar > 0.99:
            continue
        pareto["entropy"].append({
            "abstain_rate": ar,
            "critical_error_after_abstain": abstain_error_rate(preds, y_test, ab),
        })

    for gamma in [0.26, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]:
        cfg_g = copy.deepcopy(config)
        cfg_g.assurance.policy.max_set_fraction = gamma
        pipe_g = SkyCertPipeline(reasoner=reasoner, config=cfg_g, audit_path=None)
        pipe_g.calibrate(data.X_calib, data.y_calib)
        dec_g = pipe_g.run_batch(X_stream)
        pipe_g.close()
        ab_g = np.array(
            [d.kind in {DecisionKind.ABSTAIN, DecisionKind.ALERT, DecisionKind.ESCALATE}
             for d in dec_g]
        )
        ar = float(ab_g.mean())
        if ar < 0.01:
            continue
        pareto["skycert"].append({
            "gamma": gamma,
            "abstain_rate": ar,
            "critical_error_after_abstain": abstain_error_rate(preds, y_test, ab_g),
        })

    output = {
        "scenario": "distribution_shift",
        "strength": 0.8,
        "target_abstain_rate": target_abstain,
        "baselines": baselines,
        "backbones": backbones,
        "pareto": pareto,
    }

    out_path = out_dir / args.output_name
    output["seed"] = config.seed
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(safe_json(output), fh, indent=2)
    print(f"[SkyCert] wrote {out_path}")


if __name__ == "__main__":
    main()
