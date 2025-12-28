import logging
import json
import time
from typing import Dict, List
from .event_bus import AssetEventBus, TelemetryEvent
from .identity_service import DIDManager, DeviceFingerprint
from .pricing_engine import PricingEngine, DataPacket
from .asset_contract import AssetContractService

logger = logging.getLogger("AssetIntegration")

class AssetizationPipeline:
    """
    Orchestrates the full lifecycle from Flight Data -> Asset.
    Acts as a Subscriber to the AssetEventBus.
    """

    def __init__(self):
        # 1. Initialize Services
        self.did_manager = DIDManager()
        self.pricing_engine = PricingEngine()
        self.contract_service = AssetContractService()
        
        # 2. Local Buffer for Flight Missions {uav_id: [packets]}
        self.mission_buffers: Dict[str, List[DataPacket]] = {}
        
        # 3. Register Subscribers
        self.bus = AssetEventBus.get_instance()
        self.bus.subscribe("TELEMETRY_UPDATE", self.on_telemetry)
        self.bus.subscribe("MISSION_COMPLETE", self.on_mission_complete)
        
        logger.info("Assetization Pipeline Initialized.")

    def register_uav(self, uav_id: str) -> str:
        """Helper to register a UAV before flight to get a DID."""
        fingerprint = DeviceFingerprint(
            uav_id=uav_id,
            hardware_hash=f"hash_{uav_id}",
            firmware_version="v1.0",
            manufacturer_signature="DJI"
        )
        return self.did_manager.register_device(fingerprint)

    def on_telemetry(self, event: TelemetryEvent):
        """
        Step 1 & 2: Receive Data -> Sign (Identity) -> Buffer
        """
        uav_id = event.source_id
        # In a real scenario, the UAV signs its own data.
        # Here we simulate the TEE signing it based on the registered DID.
        
        # Resolve DID (Assuming uav_id mapping exists or using ID as lookup)
        # For simplicity, we assume register_uav was called and we can find the DID.
        # In this demo, we'll look up the TEE by iterating (inefficient but works for demo)
        # or we just rely on the uav_id being mapped.
        
        # Let's assume the event source_id IS the uav_id (e.g., "uav_0")
        # We need to find the DID for this uav_id.
        # We'll rely on a side-channel mapping or just find the one TEE that matches.
        
        # Hack for demo: Iterate registered DIDs to find the one with matching hardware_hash suffix
        target_did = None
        for did, tee in self.did_manager._tee_instances.items():
            # In register_uav we used f"hash_{uav_id}"
            if tee._internal_key == f"hash_{uav_id}": 
                target_did = did
                break
        
        if not target_did:
            # Auto-register if not found (for smooth demo flow)
            target_did = self.register_uav(uav_id)

        # 1. Sign Data (Identity Module)
        tee = self.did_manager.get_tee_for_device(target_did)
        priv_key = self.did_manager.get_private_key_handle(target_did)
        
        signed_payload = tee.sign_data(event.payload, priv_key)
        
        # 2. Wrap into DataPacket
        packet = DataPacket(
            source_did=target_did,
            data_type="TRAJECTORY",
            quality_score=0.95, # Mock quality
            payload=event.payload,
            timestamp=event.timestamp
        )
        
        # 3. Buffer
        if uav_id not in self.mission_buffers:
            self.mission_buffers[uav_id] = []
        self.mission_buffers[uav_id].append(packet)
        
        # print(f"[Pipeline] Buffered signed packet for {uav_id} (DID: {target_did[-6:]}...)")

    def on_mission_complete(self, event: TelemetryEvent):
        """
        Step 3 & 4: Valuation -> Tokenization -> Certificate
        """
        uav_id = event.source_id
        packets = self.mission_buffers.get(uav_id, [])
        
        if not packets:
            logger.warning(f"No packets found for mission {uav_id}")
            return

        print(f"\n=== [ASSET] Assetization Triggered for {uav_id} ===")
        print(f"Collected {len(packets)} telemetry packets.")

        # 1. Aggregate Valuation (Sum of packet values or holistic valuation)
        # For this demo, we value the last packet as the "Mission Summary" 
        # but accumulating value is also valid.
        total_value = 0.0
        last_valuation = None
        
        for p in packets:
            val_result = self.pricing_engine.evaluate(p)
            total_value += val_result.estimated_price
            last_valuation = val_result
            
        print(f"Total Mission Value: {total_value:.4f} SKY")

        # 2. Mint RWA Asset (Tokenization Module)
        # We create one NFT for the entire mission
        metadata = last_valuation.metadata
        metadata["packet_count"] = len(packets)
        metadata["total_value"] = total_value
        metadata["mission_id"] = event.event_id
        
        token_id = self.contract_service.mint_asset(
            asset_id=f"mission-{event.event_id[:8]}",
            owner_did=packets[0].source_did,
            metadata=metadata
        )

        # 3. Print Certificate
        self._print_certificate(token_id, packets[0].source_did, total_value, metadata)
        
        # Cleanup buffer
        del self.mission_buffers[uav_id]

    def _print_certificate(self, token_id, owner, value, metadata):
        cert = f"""
        ==============================================================
                     SKYNET DIGITAL ASSET CERTIFICATE               
        ==============================================================
         Asset Type:      RWA / Flight Data                         
         Token ID:        {token_id[:24]}...               
         Owner DID:       {owner}             
                                                                    
         Valuation:       {value:.4f} SKY                            
         Data Points:     {metadata['packet_count']}                       
         Quality Score:   {metadata['quality_score']}                       
         Scarcity:        {metadata['scarcity_level']}                       
                                                                    
         Blockchain Status: MINTED (Simulated)                      
        ==============================================================
        """
        print(cert)

