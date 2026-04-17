import numpy as np

from skycert.assurance.policy import (
    AssurancePolicy,
    DecisionKind,
)


def test_accept_when_confident_and_quiet():
    policy = AssurancePolicy(num_classes=4, max_set_fraction=0.75)
    probs = np.array([0.9, 0.05, 0.03, 0.02])
    mask = np.array([True, False, False, False])
    d = policy.decide(probs, mask, martingale_value=1.0, martingale_alert=False)
    assert d.kind == DecisionKind.ACCEPT
    assert d.top1 == 0


def test_abstain_on_wide_set():
    policy = AssurancePolicy(num_classes=4, max_set_fraction=0.75)
    probs = np.array([0.3, 0.25, 0.25, 0.2])
    mask = np.array([True, True, True, True])  # 4/4 = 1.0 >= 0.75
    d = policy.decide(probs, mask, martingale_value=1.0, martingale_alert=False)
    assert d.kind == DecisionKind.ABSTAIN


def test_alert_on_martingale_even_with_small_set():
    policy = AssurancePolicy(num_classes=4, max_set_fraction=0.75)
    probs = np.array([0.8, 0.1, 0.05, 0.05])
    mask = np.array([True, False, False, False])
    d = policy.decide(probs, mask, martingale_value=100.0, martingale_alert=True)
    assert d.kind == DecisionKind.ALERT


def test_escalate_when_both_conditions_trigger():
    policy = AssurancePolicy(
        num_classes=4, max_set_fraction=0.75, escalate_on_martingale=True
    )
    probs = np.array([0.3, 0.25, 0.25, 0.2])
    mask = np.array([True, True, True, True])
    d = policy.decide(probs, mask, martingale_value=100.0, martingale_alert=True)
    assert d.kind == DecisionKind.ESCALATE
