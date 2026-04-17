"""Reproduce the SkyCert main experiment.

This script mirrors Section 6 ("Experimental Setup") and Section 7
("Results and Analysis") of the ESORICS submission.

Steps:

1. Build a synthetic UAM risk dataset and split into train / calib / test.
2. Fit the neural scorer on ``X_train``.
3. Build the symbolic rule engine from the YAML config.
4. Calibrate the SkyCert conformal predictor on the clean calibration set.
5. For every configured threat, replay the test stream through
   ``SkyCertPipeline`` and measure calibration / detection / safety metrics.
6. Dump all metrics as ``outputs/metrics.json``.
7. Dump figure-ready data and audit logs under ``outputs/`` as well.
"""

from __future__ import annotations

import argparse
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
from skycert.config import SkyCertConfig, ThreatConfig
from skycert.data import apply_threat, make_uam_dataset
from skycert.metrics import (
    abstain_error_rate,
    average_set_size,
    critical_error_rate,
    detection_metrics,
    empirical_coverage,
    expected_calibration_error,
)
from skycert.pipeline import SkyCertPipeline
from skycert.utils import ensure_dir, safe_json


def _run_single_threat(
    *,
    threat: ThreatConfig,
    base_reasoner: NeuroSymbolicRiskReasoner,
    config: SkyCertConfig,
    X_test: np.ndarray,
    y_test: np.ndarray,
    audit_dir: Path,
) -> dict[str, Any]:
    rng = np.random.default_rng(config.seed + hash(threat.name) % (2**32))
    X_eval = X_test.copy()
    reasoner = base_reasoner
    change_point: int | None = None

    # Build a stream: first half IID-clean, second half under attack.
    half = X_eval.shape[0] // 2
    X_stream = X_eval.copy()
    if threat.kind != "none":
        updates = apply_threat(
            threat.kind,
            X=X_eval[half:],
            rules=config.symbolic.rules,
            strength=threat.strength,
            num_features=config.data.num_features,
            num_classes=config.data.num_classes,
            rng=rng,
        )
        if "X" in updates:
            X_stream[half:] = updates["X"]
        if "rules" in updates:
            corrupted_engine = SymbolicRuleEngine(
                updates["rules"], num_classes=config.data.num_classes
            )
            reasoner = NeuroSymbolicRiskReasoner(
                base_reasoner.neural, corrupted_engine, base_reasoner.lambda_
            )
        change_point = half

    audit_path = audit_dir / f"audit_{threat.name}.jsonl"
    pipeline = SkyCertPipeline(
        reasoner=reasoner, config=config, audit_path=audit_path
    )
    # Calibration always happens on the clean calibration set -- the
    # attacker has no access to the held-out calibration samples.
    pipeline.calibrate(_CALIB_CACHE["X"], _CALIB_CACHE["y"])

    # Drive the online pipeline sample-by-sample.
    decisions = pipeline.run_batch(X_stream)
    pipeline.close()

    # Vectorised metrics.
    probs = pipeline.predict_proba(X_stream)
    set_mask = pipeline.predict_sets(X_stream)
    preds = probs.argmax(axis=1)

    abstained = np.array(
        [d.kind in {DecisionKind.ABSTAIN, DecisionKind.ALERT, DecisionKind.ESCALATE}
         for d in decisions]
    )
    alerts = np.array([d.martingale_alert for d in decisions])
    escalations = np.array([d.kind == DecisionKind.ESCALATE for d in decisions])

    metrics = {
        "threat": {
            "name": threat.name,
            "kind": threat.kind,
            "strength": threat.strength,
        },
        "coverage": empirical_coverage(set_mask, y_test),
        "average_set_size": average_set_size(set_mask),
        "ece": expected_calibration_error(probs, y_test),
        "top1_accuracy": float((preds == y_test).mean()),
        "critical_error_rate_base": critical_error_rate(preds, y_test),
        "critical_error_rate_after_abstain": abstain_error_rate(
            preds, y_test, abstained
        ),
        "abstain_rate": float(abstained.mean()),
        "alert_rate": float(alerts.mean()),
        "escalation_rate": float(escalations.mean()),
        "detection": detection_metrics(alerts, change_point),
        "martingale_max": pipeline.monitor.max_value(),
    }
    return metrics


_CALIB_CACHE: dict[str, np.ndarray] = {}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run SkyCert main experiment")
    parser.add_argument("--config", required=True, help="path to YAML config")
    args = parser.parse_args(argv)

    config = SkyCertConfig.load(args.config)
    out_dir = ensure_dir(config.output_dir)
    audit_dir = ensure_dir(out_dir / "audit")

    data = make_uam_dataset(
        num_train=config.data.num_train,
        num_calib=config.data.num_calib,
        num_test=config.data.num_test,
        num_features=config.data.num_features,
        num_classes=config.data.num_classes,
        class_prior=config.data.class_prior,
        seed=config.seed,
    )
    _CALIB_CACHE["X"] = data.X_calib
    _CALIB_CACHE["y"] = data.y_calib

    neural = NeuralRiskScorer(
        num_classes=data.num_classes,
        l2=config.base_model.l2,
        max_iter=config.base_model.max_iter,
    ).fit(data.X_train, data.y_train)
    symbolic = SymbolicRuleEngine(
        rules=config.symbolic.rules, num_classes=data.num_classes
    )
    reasoner = NeuroSymbolicRiskReasoner(
        neural=neural, symbolic=symbolic, lambda_=config.symbolic.lambda_
    )

    all_metrics: list[dict[str, Any]] = []
    for threat in config.threats:
        print(f"[SkyCert] running threat: {threat.name} ({threat.kind})")
        metrics = _run_single_threat(
            threat=threat,
            base_reasoner=reasoner,
            config=config,
            X_test=data.X_test,
            y_test=data.y_test,
            audit_dir=audit_dir,
        )
        all_metrics.append(metrics)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(safe_json({"runs": all_metrics}), fh, indent=2)

    print(f"[SkyCert] wrote {out_dir/'metrics.json'}")
    print(f"[SkyCert] wrote audit logs under {audit_dir}")


if __name__ == "__main__":
    main()
