import numpy as np

from skycert.assurance.martingale import MartingaleMonitor, SimpleJumperBetting


def test_simple_jumper_integrates_to_one():
    f = SimpleJumperBetting(epsilon=0.92)
    # Left Riemann sum; exact integral of eps * p^(eps-1) on (0,1] is 1.
    ps = np.linspace(1e-6, 1.0, 10_000)
    vals = np.array([f(float(p)) for p in ps])
    integral = float(np.trapezoid(vals, ps))
    assert abs(integral - 1.0) < 0.01


def test_martingale_ignores_exchangeable_stream():
    np.random.seed(0)
    mon = MartingaleMonitor(threshold=50.0,
                            betting=SimpleJumperBetting(epsilon=0.92))
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, size=500)
    for s in scores:
        mon.update(float(s))
    # Under exchangeability, Ville's inequality says the max of the
    # martingale exceeds 50 with probability <= 1/50. It is extremely
    # unlikely to happen on a single uniform run with seed fixed.
    assert mon.max_value() < 50.0


def test_martingale_reacts_to_shift():
    mon = MartingaleMonitor(threshold=20.0,
                            betting=SimpleJumperBetting(epsilon=0.92),
                            seed=7)
    rng = np.random.default_rng(1)
    reference = rng.uniform(0.0, 0.2, size=500)
    mon.warm_start(reference)
    phase1 = rng.uniform(0.0, 0.2, size=200)
    phase2 = rng.uniform(0.7, 1.0, size=300)
    stream = np.concatenate([phase1, phase2])
    triggered = False
    for i, s in enumerate(stream):
        state = mon.update(float(s))
        if state["alert"] and i >= 200:
            triggered = True
            break
    assert triggered, "martingale failed to flag the injected shift"
