import hashlib
import json
import time
from typing import Dict, Any, Optional
from .tokenization import AbstractTokenizationService, AssetToken

class AssetContractService(AbstractTokenizationService):
    """
    Simulates an ERC-721 Smart Contract for Data Assets (RWA).
    Maintains an in-memory ledger of minted assets.
    """

    def __init__(self):
        # Ledger: { token_id: AssetToken }
        self._ledger: Dict[str, AssetToken] = {}
        # Owner Index: { did: [token_id, ...] }
        self._balance_of: Dict[str, list] = {}
        
        self.contract_address = "0xSkyNetAssetContractV1"

    def mint_asset(self, asset_id: str, owner_did: str, metadata: Dict[str, Any]) -> str:
        """
        Mints a new NFT representing the data asset.
        Triggers 'Transfer' event (simulated).
        """
        # 1. Generate Token ID (Hash of ID + Time + Owner)
        raw_string = f"{asset_id}{owner_did}{time.time()}"
        token_id = hashlib.sha256(raw_string.encode()).hexdigest()

        # 2. Create Asset Token Object
        new_token = AssetToken(
            token_id=token_id,
            owner_did=owner_did,
            metadata_uri=f"ipfs://{asset_id}", # Simulated IPFS link
            value=metadata.get("value", 0.0),
            status="MINTED"
        )
        # Store full metadata in the simulation for easier retrieval
        # In real solidity, this would be a JSON on IPFS
        new_token.metadata_content = metadata 

        # 3. Update State (Ledger)
        self._ledger[token_id] = new_token
        
        if owner_did not in self._balance_of:
            self._balance_of[owner_did] = []
        self._balance_of[owner_did].append(token_id)

        # 4. Log Event (Simulate Event Emission)
        # print(f"[Chain Event] Transfer(0x0, {owner_did}, {token_id})")

        return token_id

    def burn_asset(self, token_id: str) -> bool:
        """
        Destroys a token.
        """
        if token_id not in self._ledger:
            return False
        
        token = self._ledger[token_id]
        owner = token.owner_did
        
        # Update State
        del self._ledger[token_id]
        if owner in self._balance_of:
            self._balance_of[owner].remove(token_id)
            
        return True

    def transfer_asset(self, token_id: str, from_did: str, to_did: str) -> str:
        """
        Transfers ownership from one DID to another.
        """
        if token_id not in self._ledger:
            raise ValueError("Token does not exist")
        
        token = self._ledger[token_id]
        if token.owner_did != from_did:
            raise PermissionError("Sender is not the owner")

        # Update Ledger
        token.owner_did = to_did
        
        # Update Balances
        if from_did in self._balance_of:
            self._balance_of[from_did].remove(token_id)
        
        if to_did not in self._balance_of:
            self._balance_of[to_did] = []
        self._balance_of[to_did].append(token_id)

        token.status = "TRANSFERRED"
        
        return f"tx_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]}"

    def get_asset_details(self, token_id: str) -> Optional[AssetToken]:
        return self._ledger.get(token_id)

    def get_balance(self, did: str) -> int:
        return len(self._balance_of.get(did, []))

