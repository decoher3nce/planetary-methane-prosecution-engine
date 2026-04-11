"""Tests for the court-ready evidence packaging system."""

import json
from datetime import datetime, timezone

import pytest

from src.evidence.chain_of_custody import ChainOfCustody, CustodyAction
from src.evidence.packager import EvidencePackage, EvidencePackager, ManifestEntry
from src.facilities.models import (
    Facility,
    FacilityFootprint,
    FacilityType,
    InfrastructureSignature,
)
from src.fusion.engine import CorroborationLevel, FusedDetection, FusionConfidence
from src.reports.attribution import AttributionReport, ReportGenerator
from src.sensors.models import BoundingBox, Coordinate

NOW = datetime(2026, 3, 15, 14, 30, 0, tzinfo=timezone.utc)


def _make_report_and_chain():
    """Create a full attribution report with chain for testing."""
    detection = FusedDetection(
        detection_id="DET-TEST-001",
        location=Coordinate(lon=-103.72, lat=31.81),
        detection_time=NOW,
        corroboration_level=CorroborationLevel.TRIPLE,
        confidence=FusionConfidence.HIGH,
        confidence_score=0.85,
        emission_rate_kg_h=500.0,
        emission_rate_uncertainty_kg_h=75.0,
        emission_rate_source="ghgsat",
    )

    facility = Facility(
        facility_id="FAC-TEST0001",
        facility_type=FacilityType.WELL_PAD,
        name="Permian Well Pad #4117",
        operator="Test Oil Co.",
        footprint=FacilityFootprint(
            centroid=Coordinate(lon=-103.722, lat=31.811),
            bounding_box=BoundingBox(
                min_lon=-103.723, min_lat=31.810,
                max_lon=-103.721, max_lat=31.812,
            ),
            area_m2=5000,
            perimeter_m=300,
        ),
        signatures_detected=[
            InfrastructureSignature.CLEARED_PAD,
            InfrastructureSignature.CIRCULAR_TANK,
        ],
        identification_confidence=0.85,
        distance_to_plume_m=250.0,
        upwind_of_plume=True,
        epa_ghgrp_id="1234567",
        api_well_number="42-461-12345-00",
    )

    chain = ChainOfCustody(description="Permian Basin super-emitter prosecution")
    # Simulate prior chain links
    chain.add_link(
        action=CustodyAction.INGEST,
        actor="pmpe.fusion",
        description="Ingested GHGSat observation",
        input_hashes=["aaa"],
    )
    chain.add_link(
        action=CustodyAction.FUSE,
        actor="pmpe.fusion",
        description="Fused 3-sensor detection",
        input_hashes=["bbb"],
    )

    gen = ReportGenerator()
    report = gen.generate(detection, facility, chain)
    return report, chain


class TestEvidencePackager:
    def test_package_created(self):
        report, chain = _make_report_and_chain()
        packager = EvidencePackager()
        pkg = packager.package(report, chain)

        assert pkg.package_id.startswith("PKG-")
        assert pkg.case_id == chain.case_id
        assert len(pkg.artifacts) >= 4  # report, bundle, chain, manifest
        assert len(pkg.manifest) >= 4
        assert pkg.seal is not None

    def test_artifacts_present(self):
        report, chain = _make_report_and_chain()
        packager = EvidencePackager()
        pkg = packager.package(report, chain)

        assert "attribution_report.txt" in pkg.artifacts
        assert "evidence_bundle.json" in pkg.artifacts
        assert "chain_of_custody.json" in pkg.artifacts
        assert "manifest.json" in pkg.artifacts

    def test_report_rendered(self):
        report, chain = _make_report_and_chain()
        packager = EvidencePackager()
        pkg = packager.package(report, chain)

        rendered = pkg.artifacts["attribution_report.txt"].decode("utf-8")
        assert "PLANETARY METHANE PROSECUTION ENGINE" in rendered
        assert "DET-TEST-001" in rendered
        assert "Well Pad" in rendered
        assert "500" in rendered  # emission rate
        assert "VERIFIED" in rendered  # chain integrity

    def test_evidence_bundle_valid_json(self):
        report, chain = _make_report_and_chain()
        packager = EvidencePackager()
        pkg = packager.package(report, chain)

        data = json.loads(pkg.artifacts["evidence_bundle.json"])
        assert data["detection"]["detection_id"] == "DET-TEST-001"
        assert data["facility"]["facility_type"] == "well_pad"
        assert data["emission"]["rate_kg_h"] == 500.0

    def test_chain_of_custody_in_package(self):
        report, chain = _make_report_and_chain()
        packager = EvidencePackager()
        pkg = packager.package(report, chain)

        chain_data = json.loads(pkg.artifacts["chain_of_custody.json"])
        assert chain_data["case_id"] == chain.case_id
        # Should have: ingest + fuse + report + package
        assert len(chain_data["links"]) >= 3

    def test_additional_artifacts(self):
        report, chain = _make_report_and_chain()
        packager = EvidencePackager()
        pkg = packager.package(
            report, chain,
            additional_artifacts={"plume_visualization.png": b"fake-png-data"},
        )

        assert "plume_visualization.png" in pkg.artifacts
        assert pkg.artifacts["plume_visualization.png"] == b"fake-png-data"


class TestPackageSeal:
    def test_seal_valid(self):
        report, chain = _make_report_and_chain()
        packager = EvidencePackager()
        pkg = packager.package(report, chain)

        assert pkg.verify_seal() is True

    def test_seal_detects_tampering(self):
        report, chain = _make_report_and_chain()
        packager = EvidencePackager()
        pkg = packager.package(report, chain)

        # Tamper with an artifact
        pkg.artifacts["attribution_report.txt"] = b"TAMPERED CONTENT"
        # Seal should still match manifest (it's based on manifest hashes)
        # But artifact verification should fail
        errors = pkg.verify_artifacts()
        assert len(errors) > 0
        assert any("hash mismatch" in e for e in errors)

    def test_seal_detects_manifest_change(self):
        report, chain = _make_report_and_chain()
        packager = EvidencePackager()
        pkg = packager.package(report, chain)

        # Modify a manifest hash
        original_seal = pkg.seal
        pkg.manifest[0].sha256 = "0" * 64
        pkg.compute_seal()
        assert pkg.seal != original_seal


class TestPackageVerification:
    def test_all_artifacts_valid(self):
        report, chain = _make_report_and_chain()
        packager = EvidencePackager()
        pkg = packager.package(report, chain)

        errors = pkg.verify_artifacts()
        assert errors == []

    def test_missing_artifact_detected(self):
        report, chain = _make_report_and_chain()
        packager = EvidencePackager()
        pkg = packager.package(report, chain)

        # Remove an artifact
        del pkg.artifacts["evidence_bundle.json"]
        errors = pkg.verify_artifacts()
        assert any("missing from artifacts" in e for e in errors)

    def test_size_mismatch_detected(self):
        report, chain = _make_report_and_chain()
        packager = EvidencePackager()
        pkg = packager.package(report, chain)

        # Change content but keep same hash entry
        original = pkg.artifacts["attribution_report.txt"]
        pkg.artifacts["attribution_report.txt"] = original + b"extra"
        errors = pkg.verify_artifacts()
        assert len(errors) > 0


class TestPackageOutput:
    def test_to_summary(self):
        report, chain = _make_report_and_chain()
        packager = EvidencePackager()
        pkg = packager.package(report, chain)

        summary = pkg.to_summary()
        assert summary["package_id"].startswith("PKG-")
        assert summary["artifact_count"] >= 4
        assert summary["seal_valid"] is True

    def test_write_to_directory(self, tmp_path):
        report, chain = _make_report_and_chain()
        packager = EvidencePackager()
        pkg = packager.package(report, chain)

        out_dir = pkg.write_to_directory(str(tmp_path))
        from pathlib import Path
        out = Path(out_dir)

        assert out.exists()
        assert (out / "attribution_report.txt").exists()
        assert (out / "evidence_bundle.json").exists()
        assert (out / "chain_of_custody.json").exists()
        assert (out / "manifest.json").exists()

        # Verify written content matches artifact
        written = (out / "evidence_bundle.json").read_bytes()
        assert written == pkg.artifacts["evidence_bundle.json"]
