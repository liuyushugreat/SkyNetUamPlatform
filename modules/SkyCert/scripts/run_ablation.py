"""Ablation study for SkyCert.

Toggles each assurance component in turn and compares the resulting
safety/usefulness trade-off against the full SkyCert pipeline.

Ablations:
    (a) no_conformal  — use top-1 prediction only, no risk set.
    (b) no_martingale — disable drift monitor (only abstention triggers).
    (c) no_abstention — force every decision to ACCEPT.
    (d) full          — the canonical SkyCert configuration.
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
)
from skycert.pipeline import SkyCertPipeline
from skycert.utils import ensure_dir, safe_json


def _run(
    name: str,
    config: SkyCertConfig,
    neural: NeuralRiskScorer,
    symbolic: SymbolicRuleEngine,
    calib: tuple[np.ndarray, np.ndarray],
    stream: tuple[np.ndarray, np.ndarray],
    audit_path: Path,
) -> dict[str, Any]:
    reasoner = NeuroSymbolicRiskReasoner(
        neural, symbolic, lambda_=config.symbolic.lambda_
    )
    pipeline = SkyCertPipeline(
        reasoner=reasoner, config=config, audit_path=audit_path
    )
    X_calib, y_calib = calib
    X_stream, y_stream = stream
    pipeline.calibrate(X_calib, y_calib)
    decisions = pipeline.run_batch(X_stream)
    pipeline.close()

    probs = pipeline.predict_proba(X_stream)
    set_mask = pipeline.predict_sets(X_stream)
    preds = probs.argmax(axis=1)

    abstained = np.array(
        [d.kind in {DecisionKind.ABSTAIN, DecisionKind.ALERT, DecisionKind.ESCALATE}
         for d in decisions]
    )
    return {
        "variant": name,
        "coverage": empirical_coverage(set_mask, y_stream),
        "avg_set_size": average_set_size(set_mask),
        "ece": expected_calibration_error(probs, y_stream),
        "top1_accuracy": float((preds == y_stream).mean()),
        "critical_error_rate_base": critical_error_rate(preds, y_stream),
        "critical_error_rate_after_abstain": abstain_error_rate(
            preds, y_stream, abstained
        ),
        "abstain_rate": float(abstained.mean()),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run SkyCert ablation study")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-name", default="ablation.json")
    parser.add_argument("--audit-dir-name", default="audit_ablation")
    args = parser.parse_args(argv)

    config = SkyCertConfig.load(args.config)
    if args.seed is not None:
        config.seed = int(args.seed)
    out_dir = ensure_dir(config.output_dir)
    audit_dir = ensure_dir(out_dir / args.audit_dir_name)

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

    # Use a stressed stream (covariate shift) so ablations really bite.
    # Seed matches run_experiment.py so that full-SkyCert numbers are
    # identical in Table 1 and Table 2.
    threat_seed = int.from_bytes(
        hashlib.sha256(b"distribution_shift").digest()[:4], "little"
    )
    rng = np.random.default_rng(config.seed + threat_seed)
    X_stream = data.X_test.copy()
    half = X_stream.shape[0] // 2
    updates = apply_threat(
        "covariate_shift",
        X=X_stream[half:],
        strength=0.8,
        rng=rng,
    )
    X_stream[half:] = updates["X"]
    stream = (X_stream, data.y_test)
    calib = (data.X_calib, data.y_calib)

    results: list[dict[str, Any]] = []

    # (a) no conformal: set a very small alpha so q_hat ~ 1 -> sets are full,
    #     which effectively removes conformal information from decisions.
    cfg_a = copy.deepcopy(config)
    cfg_a.assurance.conformal.alpha = 0.001
    cfg_a.assurance.policy.max_set_fraction = 1.01  # never abstain from set size
    results.append(
        _run("no_conformal", cfg_a, neural, symbolic, calib, stream,
             audit_dir / "no_conformal.jsonl")
    )

    # (b) no martingale: use an infinite threshold so it never fires.
    cfg_b = copy.deepcopy(config)
    cfg_b.assurance.martingale.threshold = float("inf")
    results.append(
        _run("no_martingale", cfg_b, neural, symbolic, calib, stream,
             audit_dir / "no_martingale.jsonl")
    )

    # (c) no abstention: disable both the set-size and martingale branches.
    cfg_c = copy.deepcopy(config)
    cfg_c.assurance.policy.max_set_fraction = 1.01
    cfg_c.assurance.martingale.threshold = float("inf")
    results.append(
        _run("no_abstention", cfg_c, neural, symbolic, calib, stream,
             audit_dir / "no_abstention.jsonl")
    )

    # (d) full SkyCert.
    results.append(
        _run("full", config, neural, symbolic, calib, stream,
             audit_dir / "full.jsonl")
    )

    out_path = out_dir / args.output_name
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(safe_json({"seed": config.seed, "runs": results}), fh, indent=2)
    print(f"[SkyCert] wrote {out_path}")


if __name__ == "__main__":
    main()
