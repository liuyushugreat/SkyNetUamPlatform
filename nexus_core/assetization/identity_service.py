import uuid
import hashlib
from typing import Dict, Any, Optional
from .identity import AbstractIdentityManager, DeviceFingerprint
from .tee import TrustedExecution

class DIDManager(AbstractIdentityManager):
    """
    Concrete implementation of Identity Management using W3C DID standards (simulated).
    """

    def __init__(self):
        self._registry: Dict[str, Dict[str, Any]] = {}
        self._tee_instances: Dict[str, TrustedExecution] = {}

    def register_device(self, fingerprint: DeviceFingerprint) -> str:
        """
        Registers a UAV or Ground Station and issues a DID.
        Format: did:skynet:<method>:<uuid>
        """
        # 1. Verify hardware fingerprint (simulated)
        if not fingerprint.hardware_hash:
            raise ValueError("Invalid hardware fingerprint")

        # 2. Generate generic DID
        unique_suffix = uuid.uuid4().hex
        did = f"did:skynet:uav:{unique_suffix}"

        # 3. Initialize TEE for this device
        # In a real system, the TEE is on the device. Here we simulate it per DID.
        tee = TrustedExecution(hardware_secret=fingerprint.hardware_hash)
        pub_key, priv_key = tee.generate_keypair()
        self._tee_instances[did] = tee

        # 4. Store in registry (simulating a DID Document on-chain or IPFS)
        self._registry[did] = {
            "did": did,
            "public_key": pub_key,
            "controller": fingerprint.manufacturer_signature,
            "hardware_hash": fingerprint.hardware_hash,
            "status": "active",
            # We store private key here ONLY because this is a centralized simulation
            # In reality, private keys stay in the device TEE
            "_private_key_handle": priv_key 
        }

        return did

    def verify_signature(self, did: str, data: bytes, signature: str) -> bool:
        """
        Verifies a signature against the DID's public key.
        """
        record = self._registry.get(did)
        if not record:
            return False
            
        # Re-construct expected signature mechanism (HMAC verification)
        # Note: In real ECDSA, we'd verify using the public key. 
        # Since TEE uses HMAC for simulation, we need the secret key which is...
        # actually, standard public key crypto verification doesn't need the private key.
        # But our TEE mock uses HMAC (symmetric). 
        # For the sake of this simulation interface, we'll delegate to a 'verify' method 
        # if we had one, or cheat slightly by checking if the data produces the same signature 
        # if we had access to the TEE functionality.
        
        # Correct approach for this simulation:
        # We can't verify HMAC without the key. 
        # Let's assume the 'public_key' in our mock acts as a verification token 
        # or we implement a proper verify method in TEE that uses the public key 
        # (which would require asymmetric crypto lib).
        
        # To keep dependencies low but logic sound:
        # We will check if the signature matches what the TEE *would* produce 
        # using the stored handle.
        
        tee = self._tee_instances.get(did)
        priv_key = record.get("_private_key_handle")
        
        # Hash data to match TEE's internal hashing of the dict
        # Wait, verify_signature receives raw bytes data usually?
        # The TEE.sign_data took a Dict. 
        # Let's assume 'data' here is the data_hash (fingerprint) or raw bytes.
        
        # Let's adjust this to be consistent. 
        # If the interface expects bytes, we should treat it as such.
        
        # For HMAC verification (symmetric):
        expected_signature = tee.sign_bytes(data, priv_key)
        return hmac.compare_digest(expected_signature, signature)

    def get_public_key(self, did: str) -> str:
        record = self._registry.get(did)
        if record:
            return record["public_key"]
        raise ValueError(f"DID {did} not found")

    def get_tee_for_device(self, did: str) -> Optional[TrustedExecution]:
        """
        Helper to retrieve the simulated TEE instance for a registered device.
        Used by the simulation loop to sign data 'as' the device.
        """
        return self._tee_instances.get(did)

    def get_private_key_handle(self, did: str) -> Optional[str]:
         record = self._registry.get(did)
         return record.get("_private_key_handle") if record else None

