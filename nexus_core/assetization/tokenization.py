from abc import ABC, abstractmethod
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class AssetToken:
    """Represents an on-chain RWA token."""
    token_id: str
    owner_did: str
    metadata_uri: str
    value: float
    status: str # "MINTED", "LISTED", "SOLD"

class AbstractTokenizationService(ABC):
    """
    Interface for interacting with the Blockchain / Smart Contracts.
    Converts Valued Data into NFTs or RWA Tokens.
    """

    @abstractmethod
    def mint_asset(self, asset_id: str, owner_did: str, metadata: Dict[str, Any]) -> str:
        """
        Mints a new token and returns the transaction hash / token ID.
        """
        pass

    @abstractmethod
    def burn_asset(self, token_id: str) -> bool:
        """
        Destroys an asset (e.g., after consumption or expiry).
        """
        pass

    @abstractmethod
    def transfer_asset(self, token_id: str, from_did: str, to_did: str) -> str:
        """
        Transfers ownership.
        """
        pass

