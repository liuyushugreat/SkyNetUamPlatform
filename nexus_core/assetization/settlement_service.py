import uuid
import time
import logging
from typing import List, Dict, Any
from .settlement import AbstractSettlementManager, TransactionRecord

logger = logging.getLogger(__name__)

class SettlementManager(AbstractSettlementManager):
    """
    Handles revenue distribution and settlement for data assets.
    Implements the 70/20/10 split model.
    """

    def __init__(self):
        # Simulated Wallets: { did: balance }
        self._wallets: Dict[str, float] = {
            "did:skynet:platform:admin": 0.0,
            "did:skynet:algorithm:v1": 0.0
        }
        # Ledger for audit trails
        self._transaction_log: List[TransactionRecord] = []
        
        # Configurable Ratios
        self.ratios = {
            "provider": 0.70,
            "platform": 0.20,
            "algorithm": 0.10
        }
        
        self.PLATFORM_DID = "did:skynet:platform:admin"
        self.ALGO_DID = "did:skynet:algorithm:v1"

    def distribute_rewards(self, asset_id: str, total_amount: float, provider_did: str) -> Dict[str, Any]:
        """
        Splits the total amount and executes transfers.
        Returns a 'Settlement Credential' (Receipt).
        """
        if total_amount <= 0:
            raise ValueError("Amount must be positive")

        # 1. Calculate Splits
        provider_share = total_amount * self.ratios["provider"]
        platform_share = total_amount * self.ratios["platform"]
        algo_share = total_amount * self.ratios["algorithm"]

        timestamp = time.time()
        batch_id = uuid.uuid4().hex

        # 2. Execute Transfers (Update local state)
        self._credit_wallet(provider_did, provider_share)
        self._credit_wallet(self.PLATFORM_DID, platform_share)
        self._credit_wallet(self.ALGO_DID, algo_share)

        # 3. Log Transactions
        txs = [
            TransactionRecord(f"tx_{batch_id}_1", "EXTERNAL_MARKET", provider_did, provider_share, "SKY", timestamp),
            TransactionRecord(f"tx_{batch_id}_2", "EXTERNAL_MARKET", self.PLATFORM_DID, platform_share, "SKY", timestamp),
            TransactionRecord(f"tx_{batch_id}_3", "EXTERNAL_MARKET", self.ALGO_DID, algo_share, "SKY", timestamp)
        ]
        self._transaction_log.extend(txs)

        # 4. Generate Settlement Credential
        credential = {
            "credential_id": f"settle_{batch_id}",
            "asset_id": asset_id,
            "total_amount": total_amount,
            "currency": "SKY",
            "distribution": {
                "provider": provider_share,
                "platform": platform_share,
                "algorithm": algo_share
            },
            "timestamp": timestamp,
            "status": "COMPLETED",
            "signature": f"sig_settlement_{uuid.uuid4().hex[:8]}" # Mock signature
        }
        
        logger.info(f"Settlement completed for {asset_id}: {credential}")
        return credential

    def calculate_platform_fee(self, amount: float) -> float:
        return amount * self.ratios["platform"]

    def get_wallet_balance(self, did: str) -> float:
        return self._wallets.get(did, 0.0)

    def _credit_wallet(self, did: str, amount: float):
        if did not in self._wallets:
            self._wallets[did] = 0.0
        self._wallets[did] += amount

