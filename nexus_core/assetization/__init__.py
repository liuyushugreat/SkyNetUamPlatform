"""
SkyNet Data Assetization Layer
==============================

This package handles the full lifecycle of flight data assetization:
1. Identity: DID and Hardware Fingerprints
2. Valuation: Dynamic Pricing of Data Elements
3. Tokenization: RWA Conversion
4. Oracle: Cross-domain verification
5. Settlement: Revenue distribution

Architecture:
-------------
Uses an Event-Driven Architecture (EventBus) to decouple from the 
core flight simulation (UAVManager/Telemetry).
"""

from .event_bus import AssetEventBus
from .identity import AbstractIdentityManager, DeviceFingerprint
from .identity_service import DIDManager
from .valuation import AbstractValuationEngine, DataPacket, ValuationResult
from .pricing_engine import PricingEngine
from .tokenization import AbstractTokenizationService, AssetToken
from .asset_contract import AssetContractService
from .settlement import AbstractSettlementManager, TransactionRecord
from .settlement_service import SettlementManager
from .oracle import AbstractOracleService
from .oracle_service import DataOracle
from .tokenization import AbstractTokenizationService, AssetToken
from .oracle import AbstractOracleService
from .settlement import AbstractSettlementManager

