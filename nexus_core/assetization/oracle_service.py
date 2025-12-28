import random
import logging
from typing import Dict, Any, Optional
from .oracle import AbstractOracleService
from .settlement_service import SettlementManager

logger = logging.getLogger(__name__)

class DataOracle(AbstractOracleService):
    """
    Monitors the 'National Data Market' for purchase requests 
    and verifies external data integrity.
    """

    def __init__(self, settlement_manager: SettlementManager):
        self.settlement = settlement_manager
        # Simulated external connection state
        self.market_connection_status = "CONNECTED"

    def listen_for_requests(self, mock_event: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Simulates receiving a purchase request from the external data market.
        In production, this would be a WebSocket listener or Webhook endpoint.
        """
        # 1. Parse Request
        if not mock_event:
            return {"status": "NO_EVENT"}
            
        request_id = mock_event.get("request_id")
        asset_id = mock_event.get("asset_id")
        bid_price = mock_event.get("bid_price")
        buyer_did = mock_event.get("buyer_did")
        provider_did = mock_event.get("provider_did")

        logger.info(f"[Oracle] Received purchase request {request_id} for {asset_id} at {bid_price}")

        # 2. Verify Data Availability & Integrity (Simulated)
        # In real world, we check the On-Chain Registry or IPFS
        is_valid = self.verify_data_integrity(asset_id, "proof_mock")
        
        if is_valid:
            # 3. Trigger Settlement
            logger.info(f"[Oracle] Integrity Verified. Triggering Settlement...")
            receipt = self.settlement.distribute_rewards(
                asset_id=asset_id,
                total_amount=bid_price,
                provider_did=provider_did
            )
            return {
                "status": "SUCCESS",
                "request_id": request_id,
                "settlement_receipt": receipt
            }
        else:
            return {
                "status": "FAILED", 
                "reason": "Data Integrity Check Failed"
            }

    def verify_data_integrity(self, data_hash: str, proof: str) -> bool:
        """
        Simulates ZK-Proof verification or Signature check.
        """
        # Random failure simulation (1% chance)
        if random.random() < 0.01:
            return False
        return True

    def push_price_feed(self, asset_id: str, price: float) -> str:
        """
        Updates the on-chain oracle with the latest transaction price.
        """
        # Simulate an ETH Transaction Hash
        tx_hash = f"0x{random.randbytes(32).hex()}"
        logger.info(f"[Oracle] Pushed price {price} for {asset_id} to chain. Tx: {tx_hash}")
        return tx_hash

    def get_external_weather_data(self, location: tuple) -> Dict[str, Any]:
        return {
            "wind_speed": random.uniform(0, 15),
            "visibility": "good"
        }

