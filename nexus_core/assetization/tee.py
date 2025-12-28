import hashlib
import hmac
import secrets
import json
from typing import Dict, Any, Tuple

class TrustedExecution:
    """
    Simulates a Trusted Execution Environment (TEE) or Hardware Security Module (HSM).
    Acts as the hardware trust root for signing data at the source.
    """

    def __init__(self, hardware_secret: str = None):
        # In reality, this key is burned into silicon and cannot be extracted.
        self._internal_key = hardware_secret or secrets.token_hex(32)
    
    def generate_keypair(self) -> Tuple[str, str]:
        """
        Simulates generating a keypair inside the TEE.
        Returns (public_key, private_key_handle). 
        The private key never leaves the TEE in a real scenario.
        """
        # For simulation, we just derive a pair from the internal key
        priv_key = hashlib.sha256(f"{self._internal_key}_priv".encode()).hexdigest()
        pub_key = hashlib.sha256(f"{priv_key}_pub".encode()).hexdigest()
        return pub_key, priv_key

    def sign_data(self, data: Dict[str, Any], private_key: str) -> Dict[str, str]:
        """
        Generates a digital signature for the data packet.
        
        Args:
            data: The dictionary containing trajectory or telemetry data.
            private_key: The private key handle (simulated here as the key itself).
            
        Returns:
            Dict containing the 'hash' (SHA-256 fingerprint) and 'signature'.
        """
        # 1. Serialize data to JSON for consistent hashing
        serialized = json.dumps(data, sort_keys=True).encode()
        
        # 2. Sign bytes
        signature = self.sign_bytes(serialized, private_key)
        
        # 3. Generate Data Fingerprint (SHA-256) for the record
        data_hash = hashlib.sha256(serialized).hexdigest()

        return {
            "fingerprint": data_hash,
            "signature": signature,
            "signed_by": "TEE_SIMULATOR_V1"
        }

    def sign_bytes(self, data_bytes: bytes, private_key: str) -> str:
        """Low-level signing of raw bytes."""
        # Generate Data Fingerprint (SHA-256)
        data_hash = hashlib.sha256(data_bytes).hexdigest()
        
        # Sign the hash (HMAC-SHA256 simulation of ECDSA)
        signature = hmac.new(
            private_key.encode(), 
            data_hash.encode(), 
            hashlib.sha256
        ).hexdigest()
        return signature

    def verify_secure_boot(self) -> bool:
        """Simulates checking hardware integrity."""
        return True

