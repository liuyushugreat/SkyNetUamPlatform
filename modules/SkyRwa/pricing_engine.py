"""
Legacy pricing engine (Phase-1 compatibility layer).

.. deprecated::
    ``PricingEngine`` is a Phase-1 stub that returns a hard-coded price.
    It is kept only so that existing imports do not break.

    For new code, use:

    - ``RuleBasedValuationEngine`` for deterministic, explainable pricing
    - ``NeuralValuationAdapter`` for neural-network-assisted pricing
    - ``CongestionPricingModel`` (in ``economics.pricing``) for voxel-level
      congestion pricing

TODO(cleanup): remove this file once all callers migrate to the V2 pipeline.
FIXME(stub): ``calculate_price`` always returns 10.0 — this is intentionally
non-functional; use ``RuleBasedValuationEngine.evaluate()`` instead.
"""


class PricingEngine:
    """Legacy pricing entry-point.  Prefer ``RuleBasedValuationEngine``."""

    def __init__(self) -> None:
        pass

    def calculate_price(self, context: dict) -> float:
        # FIXME(stub): placeholder — always returns 10.0.
        # Migrate callers to RuleBasedValuationEngine.evaluate().
        return 10.0
