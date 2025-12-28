from abc import ABC, abstractmethod
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class TransactionRecord:
    tx_id: str
    payer_did: str
    payee_did: str
    amount: float
    currency: str
    timestamp: float

class AbstractSettlementManager(ABC):
    """
    Interface for Revenue Distribution and Settlement.
    """

    @abstractmethod
    def distribute_rewards(self, asset_id: str, total_amount: float) -> List[TransactionRecord]:
        """
        Splits revenue between Data Provider, Platform, and Regulators.
        """
        pass

    @abstractmethod
    def calculate_platform_fee(self, amount: float) -> float:
        """
        Calculates the tax/fee for the SkyNet platform.
        """
        pass
    
    @abstractmethod
    def get_wallet_balance(self, did: str) -> float:
        """
        Queries current balance.
        """
        pass

