from abc import ABC, abstractmethod
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class DeviceFingerprint:
    """Hardware security metadata for a UAV."""
    uav_id: str
    hardware_hash: str     # PUF (Physical Unclonable Function) or Serial Hash
    firmware_version: str
    manufacturer_signature: str

class AbstractIdentityManager(ABC):
    """
    Interface for DID (Decentralized Identity) Management.
    Handles binding of physical devices to on-chain identities.
    """

    @abstractmethod
    def register_device(self, fingerprint: DeviceFingerprint) -> str:
        """
        Registers a new device and returns its DID (e.g., did:skynet:123...).
        """
        pass

    @abstractmethod
    def verify_signature(self, did: str, data: bytes, signature: str) -> bool:
        """
        Verifies that data came from the specific hardware device.
        """
        pass

    @abstractmethod
    def get_public_key(self, did: str) -> str:
        """
        Retrieves public key for a DID.
        """
        pass

