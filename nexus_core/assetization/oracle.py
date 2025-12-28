from abc import ABC, abstractmethod
from typing import Dict, Any, List

class AbstractOracleService(ABC):
    """
    Interface for Cross-Domain Data Verification.
    Acts as a bridge between off-chain flight data and on-chain contracts.
    """

    @abstractmethod
    def verify_data_integrity(self, data_hash: str, proof: str) -> bool:
        """
        Verifies that data hasn't been tampered with (using ZK proofs or signatures).
        """
        pass

    @abstractmethod
    def push_price_feed(self, asset_id: str, price: float) -> str:
        """
        Pushes the calculated price to an on-chain Oracle Contract.
        """
        pass

    @abstractmethod
    def get_external_weather_data(self, location: tuple) -> Dict[str, Any]:
        """
        Example of fetching external data to validate flight conditions.
        """
        pass

