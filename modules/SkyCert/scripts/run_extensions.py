"""Extended analyses supporting the paper's appendix:

* lambda_drift sweep: varies the weight of the input-drift term in the
  hybrid nonconformity score and reports coverage / avg_set / critical
  miss rate.
* attack-strength sweeps: varies beta3 (feature_attack) and beta4
  (covariate_shift) over a small grid and reports how safety degrades.
* failure cases: emits three representative audit JSON snippets from the
  distribution_shift scenario that show what happened on a CRITICAL miss
  that SkyCert did not catch and on a CRITICAL miss that SkyCert
  successfully abstained on.
* MLP backbone replication: reruns the three key scenarios with a
  two-layer MLP backbone to support the model-agnostic claim.
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
    expected_calibration_error,
    fp_fn_critical,
    per_class_coverage,
)
from skycert.pipeline import SkyCertPipeline
from skycert.utils import ensure_dir, safe_json


# -----------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------
def _build_reasoner(config: SkyCertConfig, data) -> NeuroSymbolicRiskReasoner:
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
    return NeuroSymbolicRiskReasoner(
        neural=neural, symbolic=symbolic, lambda_=config.symbolic.lambda_
    )


def _threat_stream(name: str, config: SkyCertConfig, X_test: np.ndarray,
                   strength: float) -> tuple[np.ndarray, int | None]:
    threat_seed = int.from_bytes(
        hashlib.sha256(name.encode("utf-8")).digest()[:4], "little"
    )
    rng = np.random.default_rng(config.seed + threat_seed)
    X_stream = X_test.copy()
    half = X_stream.shape[0] // 2
    kind = {
        "distribution_shift": "covariate_shift",
        "input_manipulate": "feature_attack",
    }[name]
    updates = apply_threat(
        kind,
        X=X_stream[half:],
        strength=strength,
        num_features=config.data.num_features,
        num_classes=config.data.num_classes,
        rng=rng,
    )
    if "X" in updates:
        X_stream[half:] = updates["X"]
    return X_stream, half


def _run_pipeline(
    config: SkyCertConfig,
    reasoner: NeuroSymbolicRiskReasoner,
    X_calib, y_calib, X_stream, y_stream,
) -> dict[str, Any]:
    pipe = SkyCertPipeline(reasoner=reasoner, config=config, audit_path=None)
    pipe.calibrate(X_calib, y_calib)
    decisions = pipe.run_batch(X_stream)
    pipe.close()
    probs = pipe.predict_proba(X_stream)
    set_mask = pipe.predict_sets(X_stream)
    preds = probs.argmax(axis=1)
    abstained = np.array(
        [d.kind in {DecisionKind.ABSTAIN, DecisionKind.ALERT, DecisionKind.ESCALATE}
         for d in decisions]
    )
    num_classes = config.data.num_classes
    return {
        "coverage": empirical_coverage(set_mask, y_stream),
        "per_class_coverage": per_class_coverage(set_mask, y_stream, num_classes),
        "avg_set_size": average_set_size(set_mask),
        "ece": expected_calibration_error(probs, y_stream),
        "top1_accuracy": float((preds == y_stream).mean()),
        "critical_error_rate_base": critical_error_rate(preds, y_stream),
        "critical_error_rate_after_abstain":
            abstain_error_rate(preds, y_stream, abstained),
        "fp_fn": fp_fn_critical(preds, y_stream, abstained),
        "abstain_rate": float(abstained.mean()),
        "martingale_max": pipe.monitor.max_value(),
    }


# -----------------------------------------------------------------------
# Lambda sweep
# -----------------------------------------------------------------------
def lambda_sweep(config: SkyCertConfig, data) -> list[dict[str, Any]]:
    reasoner = _build_reasoner(config, data)
    X_stream, _ = _threat_stream(
        "distribution_shift", config, data.X_test, strength=0.8
    )
    results: list[dict[str, Any]] = []
    for lam in [0.3, 0.5, 1.0, 2.0, 3.0]:
        cfg = copy.deepcopy(config)
        cfg.assurance.martingale.lambda_drift = lam
        out = _run_pipeline(cfg, reasoner, data.X_calib, data.y_calib,
                            X_stream, data.y_test)
        out["lambda_drift"] = lam
        results.append(out)
    return results


# -----------------------------------------------------------------------
# Attack-strength sweep (beta3 = feature_attack, beta4 = covariate_shift)
# -----------------------------------------------------------------------
def attack_strength_sweep(config: SkyCertConfig, data) -> dict[str, Any]:
    reasoner = _build_reasoner(config, data)

    beta3_points = [0.05, 0.10, 0.15, 0.20, 0.25]
    beta4_points = [0.2, 0.4, 0.6, 0.8, 1.0]

    sweep_beta3: list[dict[str, Any]] = []
    for b in beta3_points:
        X_stream, _ = _threat_stream(
            "input_manipulate", config, data.X_test, strength=b
        )
        out = _run_pipeline(config, reasoner, data.X_calib, data.y_calib,
                            X_stream, data.y_test)
        out["strength"] = b
        sweep_beta3.append(out)

    sweep_beta4: list[dict[str, Any]] = []
    for b in beta4_points:
        X_stream, _ = _threat_stream(
            "distribution_shift", config, data.X_test, strength=b
        )
        out = _run_pipeline(config, reasoner, data.X_calib, data.y_calib,
                            X_stream, data.y_test)
        out["strength"] = b
        sweep_beta4.append(out)

    return {"beta3": sweep_beta3, "beta4": sweep_beta4}


# -----------------------------------------------------------------------
# Failure cases (read from the multi-seed baseline seed)
# -----------------------------------------------------------------------
def failure_cases(config: SkyCertConfig, data, n_examples: int = 3) -> list[dict[str, Any]]:
    reasoner = _build_reasoner(config, data)
    X_stream, change_point = _threat_stream(
        "distribution_shift", config, data.X_test, strength=0.8
    )
    pipe = SkyCertPipeline(reasoner=reasoner, config=config, audit_path=None)
    pipe.calibrate(data.X_calib, data.y_calib)
    decisions = pipe.run_batch(X_stream)
    pipe.close()

    probs = pipe.predict_proba(X_stream)
    preds = probs.argmax(axis=1)
    y = data.y_test
    # Collect: (a) true CRITICAL missed without abstention, (b) CRITICAL
    # correctly abstained on, (c) CRITICAL escalated.
    buckets = {"missed_unabstained": [], "missed_abstained": [], "escalated": []}
    for i, d in enumerate(decisions):
        if y[i] != 3:
            continue
        is_miss = preds[i] != 3
        abst = d.kind in {DecisionKind.ABSTAIN, DecisionKind.ALERT,
                          DecisionKind.ESCALATE}
        if d.kind == DecisionKind.ESCALATE:
            buckets["escalated"].append(i)
        elif is_miss and not abst:
            buckets["missed_unabstained"].append(i)
        elif is_miss and abst:
            buckets["missed_abstained"].append(i)

    # Pick one representative from each bucket (the first one available).
    picks: list[int] = []
    for key in ("missed_unabstained", "missed_abstained", "escalated"):
        if buckets[key]:
            picks.append(buckets[key][0])
        if len(picks) >= n_examples:
            break

    # Build a compact audit-style record for each pick.
    records: list[dict[str, Any]] = []
    for idx in picks:
        d = decisions[idx]
        trace = reasoner.symbolic.trace(X_stream[idx])
        records.append({
            "sample_id": int(idx),
            "after_change_point": bool(
                change_point is not None and idx >= change_point
            ),
            "true_label": int(y[idx]),
            "predicted_label": int(preds[idx]),
            "probability_vector": [float(p) for p in probs[idx]],
            "prediction_set": d.prediction_set,
            "set_size": d.set_size,
            "martingale_value": float(d.martingale),
            "decision_kind": str(d.kind.value),
            "reasons": d.reasons,
            "firing_rules": trace,
            "commentary": _commentary(idx, change_point, preds[idx], y[idx], d),
        })
    return records


def _commentary(idx: int, cp: int | None, pred: int, true: int, d) -> str:
    after = (cp is not None and idx >= cp)
    head = ("post-shift operation" if after else "pre-shift operation")
    kind = d.kind.value
    if kind == "ESCALATE":
        return (f"{head}: CRITICAL ground truth predicted as class {pred}, "
                f"but set ambiguity AND martingale threshold triggered "
                "simultaneously, so the decision was escalated.")
    if kind in ("ABSTAIN", "ALERT"):
        return (f"{head}: CRITICAL ground truth predicted as class {pred}; "
                f"SkyCert abstained ({kind}) because set size or martingale "
                "indicated epistemic uncertainty.")
    return (f"{head}: CRITICAL ground truth predicted as class {pred} and "
            "SkyCert did NOT abstain. This is a residual safety risk: the "
            "top-1 prediction was confident and the drift monitor had not "
            "yet crossed threshold at this particular step.")


# -----------------------------------------------------------------------
# MLP backbone replication
# -----------------------------------------------------------------------
def mlp_backbone(config: SkyCertConfig, data) -> list[dict[str, Any]]:
    cfg = copy.deepcopy(config)
    cfg.base_model.type = "mlp"
    cfg.base_model.hidden = 64
    cfg.base_model.max_iter = 300
    # Re-fit reasoner with MLP backbone.
    reasoner = _build_reasoner(cfg, data)

    results: list[dict[str, Any]] = []
    scenarios = [
        ("clean", "none", 0.0),
        ("input_manipulate", "feature_attack", 0.15),
        ("distribution_shift", "covariate_shift", 0.8),
    ]
    for name, kind, strength in scenarios:
        threat_seed = int.from_bytes(
            hashlib.sha256(name.encode("utf-8")).digest()[:4], "little"
        )
        rng = np.random.default_rng(cfg.seed + threat_seed)
        X_stream = data.X_test.copy()
        half = X_stream.shape[0] // 2
        if kind != "none":
            updates = apply_threat(
                kind, X=X_stream[half:], strength=strength,
                num_features=cfg.data.num_features,
                num_classes=cfg.data.num_classes, rng=rng,
            )
            if "X" in updates:
                X_stream[half:] = updates["X"]
        out = _run_pipeline(
            cfg, reasoner, data.X_calib, data.y_calib, X_stream, data.y_test
        )
        out["scenario"] = name
        out["strength"] = strength
        results.append(out)
    return results


# -----------------------------------------------------------------------
# Entry
# -----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run SkyCert extension experiments")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
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

    print("[extensions] lambda sweep ...")
    lam = lambda_sweep(config, data)
    print("[extensions] attack strength sweep ...")
    strength = attack_strength_sweep(config, data)
    print("[extensions] failure cases ...")
    fail = failure_cases(config, data)
    print("[extensions] MLP backbone replication ...")
    mlp = mlp_backbone(config, data)

    payload = {
        "seed": config.seed,
        "lambda_sweep": lam,
        "attack_strength_sweep": strength,
        "failure_cases": fail,
        "mlp_backbone": mlp,
    }
    out_path = out_dir / "extensions.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(safe_json(payload), fh, indent=2)
    print(f"[extensions] wrote {out_path}")


if __name__ == "__main__":
    main()
