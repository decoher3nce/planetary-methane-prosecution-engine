"""
Chain-of-Custody Evidence System.

Implements a tamper-evident, cryptographically linked chain of custody
for satellite observations. Each processing step is recorded as a
ChainLink with a hash that incorporates the previous link's hash,
creating an immutable audit trail from raw sensor data to final
prosecution evidence.

This is modeled after digital forensics best practices (NIST SP 800-86)
and adapted for remote-sensing evidence workflows.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class CustodyAction(str, Enum):
    """Actions recorded in the chain of custody."""

    INGEST = "ingest"  # Raw data acquired from provider
    VALIDATE = "validate"  # Data integrity / quality check passed
    COREGISTER = "coregister"  # Spatial alignment between sensors
    FUSE = "fuse"  # Multi-sensor data fusion
    QUANTIFY = "quantify"  # Emission rate quantification
    ATTRIBUTE = "attribute"  # Source attribution to facility
    CORROBORATE = "corroborate"  # Cross-sensor corroboration
    REPORT = "report"  # Evidence report generated
    PACKAGE = "package"  # Court-ready package assembled
    EXPORT = "export"  # Data exported for legal submission


class ChainLink(BaseModel):
    """A single link in the chain of custody.

    Each link records who did what, when, to which data, and
    cryptographically chains to the previous link.
    """

    link_id: str = Field(default_factory=lambda: uuid4().hex[:16])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    action: CustodyAction
    actor: str  # System component or human operator identifier
    description: str
    input_hashes: list[str]  # SHA-256 hashes of input data
    output_hash: Optional[str] = None  # SHA-256 hash of output data
    previous_link_hash: Optional[str] = None  # Hash of preceding ChainLink
    metadata: dict = Field(default_factory=dict)
    link_hash: Optional[str] = None  # This link's own hash (computed)

    def compute_link_hash(self) -> str:
        """Compute this link's hash incorporating all fields + previous link."""
        payload = {
            "link_id": self.link_id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action.value,
            "actor": self.actor,
            "description": self.description,
            "input_hashes": sorted(self.input_hashes),
            "output_hash": self.output_hash,
            "previous_link_hash": self.previous_link_hash,
            "metadata": self.metadata,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        self.link_hash = hashlib.sha256(canonical.encode()).hexdigest()
        return self.link_hash


class ChainOfCustody(BaseModel):
    """Complete chain of custody for a prosecution case.

    Maintains an ordered list of ChainLinks forming a tamper-evident
    record from sensor acquisition through evidence packaging.
    """

    case_id: str = Field(default_factory=lambda: f"PMPE-{uuid4().hex[:8].upper()}")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    description: str = ""
    links: list[ChainLink] = Field(default_factory=list)

    def add_link(
        self,
        action: CustodyAction,
        actor: str,
        description: str,
        input_hashes: list[str],
        output_hash: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> ChainLink:
        """Add a new link to the chain, automatically chaining to the previous."""
        previous_hash = self.links[-1].link_hash if self.links else None

        link = ChainLink(
            action=action,
            actor=actor,
            description=description,
            input_hashes=input_hashes,
            output_hash=output_hash,
            previous_link_hash=previous_hash,
            metadata=metadata or {},
        )
        link.compute_link_hash()
        self.links.append(link)
        return link

    def verify_integrity(self) -> tuple[bool, list[str]]:
        """Verify the entire chain's cryptographic integrity.

        Returns (is_valid, list_of_errors).
        """
        errors: list[str] = []

        if not self.links:
            return True, []

        # First link should have no previous
        if self.links[0].previous_link_hash is not None:
            errors.append("First link has a previous_link_hash (should be None)")

        for i, link in enumerate(self.links):
            # Recompute and verify each link's own hash
            stored_hash = link.link_hash
            recomputed = link.compute_link_hash()
            if stored_hash != recomputed:
                errors.append(
                    f"Link {i} ({link.link_id}): hash mismatch — "
                    f"stored={stored_hash}, recomputed={recomputed}"
                )

            # Verify chain linkage
            if i > 0:
                expected_prev = self.links[i - 1].link_hash
                if link.previous_link_hash != expected_prev:
                    errors.append(
                        f"Link {i} ({link.link_id}): previous_link_hash mismatch — "
                        f"expected={expected_prev}, got={link.previous_link_hash}"
                    )

        return len(errors) == 0, errors

    def get_audit_trail(self) -> list[dict]:
        """Return a human-readable audit trail."""
        trail = []
        for i, link in enumerate(self.links):
            trail.append({
                "step": i + 1,
                "timestamp": link.timestamp.isoformat(),
                "action": link.action.value,
                "actor": link.actor,
                "description": link.description,
                "link_hash": link.link_hash[:16] + "...",
                "chain_valid": (
                    link.previous_link_hash == self.links[i - 1].link_hash
                    if i > 0
                    else link.previous_link_hash is None
                ),
            })
        return trail

    def to_json(self) -> str:
        """Serialize the full chain to JSON for archival."""
        return self.model_dump_json(indent=2)
