"""Tests for the chain-of-custody evidence system."""

import json

from src.evidence.chain_of_custody import (
    ChainLink,
    ChainOfCustody,
    CustodyAction,
)


class TestChainLink:
    def test_compute_hash(self):
        link = ChainLink(
            action=CustodyAction.INGEST,
            actor="pmpe.sensors.tanager1",
            description="Ingested Tanager-1 scene TAN-2026-001",
            input_hashes=["abc123"],
        )
        h = link.compute_link_hash()
        assert len(h) == 64
        assert link.link_hash == h

    def test_hash_deterministic(self):
        kwargs = dict(
            link_id="test-link-001",
            action=CustodyAction.VALIDATE,
            actor="pmpe.validation",
            description="Validated data integrity",
            input_hashes=["abc123"],
        )
        link1 = ChainLink(**kwargs)
        link2 = ChainLink(**kwargs)
        # Same timestamp needed for determinism
        link2.timestamp = link1.timestamp
        assert link1.compute_link_hash() == link2.compute_link_hash()


class TestChainOfCustody:
    def _build_chain(self) -> ChainOfCustody:
        chain = ChainOfCustody(description="Test case: Permian Basin super-emitter")
        chain.add_link(
            action=CustodyAction.INGEST,
            actor="pmpe.sensors.tanager1",
            description="Ingested Tanager-1 hyperspectral scene",
            input_hashes=["aaa111"],
            output_hash="bbb222",
        )
        chain.add_link(
            action=CustodyAction.INGEST,
            actor="pmpe.sensors.ghgsat",
            description="Ingested GHGSat point-source observation",
            input_hashes=["ccc333"],
            output_hash="ddd444",
        )
        chain.add_link(
            action=CustodyAction.FUSE,
            actor="pmpe.fusion.engine",
            description="Fused Tanager-1 + GHGSat observations",
            input_hashes=["bbb222", "ddd444"],
            output_hash="eee555",
        )
        return chain

    def test_chain_builds(self):
        chain = self._build_chain()
        assert len(chain.links) == 3
        assert chain.links[0].previous_link_hash is None
        assert chain.links[1].previous_link_hash == chain.links[0].link_hash
        assert chain.links[2].previous_link_hash == chain.links[1].link_hash

    def test_verify_integrity_valid(self):
        chain = self._build_chain()
        valid, errors = chain.verify_integrity()
        assert valid is True
        assert errors == []

    def test_verify_integrity_tampered(self):
        chain = self._build_chain()
        # Tamper with a link's description
        chain.links[1].description = "TAMPERED"
        valid, errors = chain.verify_integrity()
        assert valid is False
        assert len(errors) >= 1

    def test_audit_trail(self):
        chain = self._build_chain()
        trail = chain.get_audit_trail()
        assert len(trail) == 3
        assert trail[0]["action"] == "ingest"
        assert trail[0]["chain_valid"] is True
        assert trail[2]["action"] == "fuse"

    def test_serialization(self):
        chain = self._build_chain()
        j = chain.to_json()
        data = json.loads(j)
        assert data["case_id"].startswith("PMPE-")
        assert len(data["links"]) == 3

    def test_empty_chain_valid(self):
        chain = ChainOfCustody()
        valid, errors = chain.verify_integrity()
        assert valid is True
