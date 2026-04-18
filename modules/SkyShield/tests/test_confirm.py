from __future__ import annotations

from skyshield.tracker.confirm import MofNConfirmer


def test_requires_m_hits_before_confirming():
    c = MofNConfirmer((3, 5))
    assert not c.is_confirmed(1)
    c.observe(1, 10.0)
    c.observe(1, 20.0)
    assert not c.is_confirmed(1)
    c.observe(1, 30.0)
    assert c.is_confirmed(1)


def test_confirmation_latches():
    c = MofNConfirmer((2, 3))
    c.observe(1, 0.0); c.observe(1, 10.0)
    assert c.is_confirmed(1)
    # Adding stale observations should not un-confirm.
    c.observe(1, 100.0)
    assert c.is_confirmed(1)
