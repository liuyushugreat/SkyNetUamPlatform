"""Real cryptographic signing for flight evidence using Ed25519.

Replaces the placeholder signature mechanism with actual Ed25519
signatures via the ``cryptography`` library.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from typing import Optional, Tuple

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from ..models.evidence import FlightEvidencePackage


class Ed25519Signer:
    """Sign and verify evidence digests using Ed25519."""

    def __init__(
        self,
        private_key: Optional[Ed25519PrivateKey] = None,
        signer_id: str = "system",
    ):
        self._private_key = private_key or Ed25519PrivateKey.generate()
        self._public_key = self._private_key.public_key()
        self.signer_id = signer_id

    @property
    def public_key(self) -> Ed25519PublicKey:
        return self._public_key

    def public_key_b64(self) -> str:
        from cryptography.hazmat.primitives.serialization import (
            Encoding,
            PublicFormat,
        )
        raw = self._public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)
        return base64.b64encode(raw).decode()

    def sign_evidence(self, evidence: FlightEvidencePackage) -> FlightEvidencePackage:
        """Sign the evidence digest and populate signature fields.

        Returns the same evidence instance with updated signature metadata.
        """
        if not evidence.digest_hash:
            raise ValueError("Evidence must have a digest_hash before signing")
        sig_bytes = self._private_key.sign(evidence.digest_hash.encode("utf-8"))
        evidence.signature = base64.b64encode(sig_bytes).decode()
        evidence.signed_by = self.signer_id
        evidence.signed_at = datetime.now(UTC)
        return evidence

    def verify_evidence(self, evidence: FlightEvidencePackage) -> bool:
        """Verify the signature on an evidence package."""
        if not evidence.signature or not evidence.digest_hash:
            return False
        try:
            sig_bytes = base64.b64decode(evidence.signature)
            self._public_key.verify(sig_bytes, evidence.digest_hash.encode("utf-8"))
            return True
        except Exception:
            return False

    @staticmethod
    def generate_keypair(signer_id: str = "system") -> "Ed25519Signer":
        """Create a new signer with a fresh key pair."""
        return Ed25519Signer(signer_id=signer_id)
