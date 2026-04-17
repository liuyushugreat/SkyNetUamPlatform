import numpy as np

from skycert.assurance.conformal import ConformalRiskSet


def _fake_probs(n: int, k: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((n, k))
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(k, p=p) for p in probs])
    return probs, y


def test_conformal_coverage_marginal():
    probs, y = _fake_probs(2000, 4, seed=1)
    calib_probs, calib_y = probs[:1000], y[:1000]
    test_probs, test_y = probs[1000:], y[1000:]
    for score in ("aps", "lac"):
        cp = ConformalRiskSet(alpha=0.1, score=score)
        cp.calibrate(calib_probs, calib_y)
        cov = cp.coverage(test_probs, test_y)
        # Marginal coverage is 1-alpha ± O(1/sqrt(n_cal)); use a loose bound.
        assert 0.8 <= cov <= 1.0, f"{score}: coverage={cov}"


def test_conformal_sets_include_top1():
    probs, y = _fake_probs(500, 4, seed=2)
    cp = ConformalRiskSet(alpha=0.5, score="aps")
    cp.calibrate(probs[:250], y[:250])
    mask = cp.predict_sets(probs[250:])
    top1 = np.argmax(probs[250:], axis=1)
    assert mask[np.arange(mask.shape[0]), top1].all()
