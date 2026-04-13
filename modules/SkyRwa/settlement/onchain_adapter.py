"""On-chain adapter — abstract interface for blockchain registration.

Provides a protocol-level contract for linking asset units to an on-chain
registry.  The default :class:`OnChainAdapter` is a **no-op stub** that logs
calls; real implementations (e.g. using ``web3.py`` or Hyperledger) should
subclass and override.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..models.asset_unit import FlightAssetUnit

logger = logging.getLogger(__name__)


class OnChainAdapter(ABC):
    """Protocol for registering assets and settling revenue on-chain."""

    @abstractmethod
    def register_asset(self, unit: FlightAssetUnit) -> str:
        """Register the asset unit and return a transaction / receipt id."""
        ...

    @abstractmethod
    def mint_receipt(self, unit: FlightAssetUnit) -> str:
        """Mint a receipt token / NFT for the asset and return its id."""
        ...

    @abstractmethod
    def settle_revenue(
        self,
        unit: FlightAssetUnit,
        usage_id: str,
        amount: float,
    ) -> str:
        """Record a settlement on-chain and return a tx id."""
        ...


class NoOpOnChainAdapter(OnChainAdapter):
    """Stub adapter that logs operations without touching any real chain."""

    def register_asset(self, unit: FlightAssetUnit) -> str:
        tx = f"noop-register-{unit.asset_unit_id[:12]}"
        logger.info("register_asset (no-op): %s → %s", unit.asset_unit_id, tx)
        return tx

    def mint_receipt(self, unit: FlightAssetUnit) -> str:
        tx = f"noop-mint-{unit.asset_unit_id[:12]}"
        logger.info("mint_receipt (no-op): %s → %s", unit.asset_unit_id, tx)
        return tx

    def settle_revenue(
        self,
        unit: FlightAssetUnit,
        usage_id: str,
        amount: float,
    ) -> str:
        tx = f"noop-settle-{usage_id[:12]}"
        logger.info(
            "settle_revenue (no-op): asset=%s usage=%s amount=%.4f → %s",
            unit.asset_unit_id,
            usage_id,
            amount,
            tx,
        )
        return tx
