"""Tests for automated attribution report generation."""

from datetime import datetime, timezone

import json
import pytest

from src.evidence.chain_of_custody import ChainOfCustody, CustodyAction
from src.facilities.models import (
    Facility,
    FacilityFootprint,
    FacilityType,
    InfrastructureSignature,
)
from src.fusion.engine import (
    CorroborationLevel,
    FusedDetection,
    FusionConfidence,
)
from src.reports.attribution import (
    AttributionReport,
    EmissionQuantification,
    RegulatoryContext,
    ReportGenerator,
    SeverityLevel,
    ViolationType,
)
from src.sensors.models import BoundingBox, Coordinate

NOW = datetime(2026, 3, 15, 14, 30, 0, tzinfo=timezone.utc)
PLUME_CENTER = Coordinate(lon=-103.72, lat=31.81)


def _make_detection(rate: float = 500.0, uncertainty: float = 75.0) -> FusedDetection:
    return FusedDetection(
        detection_id="DET-TEST-001",
        location=PLUME_CENTER,
        detection_time=NOW,
        corroboration_level=CorroborationLevel.TRIPLE,
        confidence=FusionConfidence.HIGH,
        confidence_score=0.85,
        emission_rate_kg_h=rate,
        emission_rate_uncertainty_kg_h=uncertainty,
        emission_rate_source="ghgsat",
    )


def _make_facility(
    ftype: FacilityType = FacilityType.WELL_PAD,
) -> Facility:
    return Facility(
        facility_id="FAC-TEST0001",
        facility_type=ftype,
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


# ── Emission Quantification ─────────────────────────────────────────


class TestEmissionQuantification:
    def test_from_detection(self):
        det = _make_detection(rate=500.0, uncertainty=75.0)
        eq = EmissionQuantification.from_detection(det)

        assert eq.rate_kg_h == 500.0
        assert eq.rate_uncertainty_kg_h == 75.0
        # 500 kg/hr * 8760 hr/yr / 1000 = 4380 tonnes/yr
        assert eq.rate_tonnes_year == 4380.0
        # 4380 * 80 (GWP-20) = 350,400
        assert eq.rate_co2e_tonnes_year == 350_400.0
        assert eq.gwp_basis == "GWP-20"

    def test_gwp100(self):
        det = _make_detection(rate=500.0)
        eq = EmissionQuantification.from_detection(det, gwp_factor=28.0, gwp_basis="GWP-100")

        assert eq.gwp_basis == "GWP-100"
        assert eq.rate_co2e_tonnes_year == pytest.approx(4380.0 * 28.0, rel=0.01)

    def test_zero_emission(self):
        det = _make_detection(rate=0.0, uncertainty=0.0)
        eq = EmissionQuantification.from_detection(det)
        assert eq.rate_tonnes_year == 0.0
        assert eq.rate_co2e_tonnes_year == 0.0


# ── Regulatory Context ──────────────────────────────────────────────


class TestRegulatoryContext:
    def test_super_emitter_well_pad(self):
        eq = EmissionQuantification.from_detection(_make_detection(rate=500.0))
        ctx = RegulatoryContext.assess(eq, FacilityType.WELL_PAD)

        assert ctx.severity == SeverityLevel.MAJOR
        assert ctx.qualifies_as_super_emitter is True
        assert ViolationType.SUPER_EMITTER_RESPONSE in ctx.applicable_violations
        assert ViolationType.EPA_SUBPART_W in ctx.applicable_violations
        assert ViolationType.EPA_OOOO in ctx.applicable_violations
        assert ViolationType.CLEAN_AIR_ACT in ctx.applicable_violations

    def test_critical_severity(self):
        eq = EmissionQuantification.from_detection(_make_detection(rate=3000.0))
        ctx = RegulatoryContext.assess(eq, FacilityType.COMPRESSOR_STATION)

        assert ctx.severity == SeverityLevel.CRITICAL
        assert ctx.estimated_penalty_usd is not None
        assert ctx.estimated_penalty_usd > 0

    def test_minor_emission(self):
        eq = EmissionQuantification.from_detection(_make_detection(rate=50.0))
        ctx = RegulatoryContext.assess(eq, FacilityType.WELL_PAD)

        assert ctx.severity == SeverityLevel.MINOR
        assert ctx.qualifies_as_super_emitter is False

    def test_landfill_violation(self):
        eq = EmissionQuantification.from_detection(_make_detection(rate=200.0))
        ctx = RegulatoryContext.assess(eq, FacilityType.LANDFILL)

        assert ViolationType.CLEAN_AIR_ACT in ctx.applicable_violations
        assert ctx.qualifies_as_super_emitter is True

    def test_penalty_estimation(self):
        eq = EmissionQuantification.from_detection(_make_detection(rate=600.0))
        ctx = RegulatoryContext.assess(eq, FacilityType.WELL_PAD)

        # $65k/day * 30 days * N violations
        assert ctx.estimated_penalty_usd is not None
        assert ctx.estimated_penalty_usd >= 65_000 * 30

    def test_exceeds_reporting_threshold(self):
        # 500 kg/hr → 4380 t/yr → 350,400 t CO2e/yr (GWP-20) > 25,000
        eq = EmissionQuantification.from_detection(_make_detection(rate=500.0))
        ctx = RegulatoryContext.assess(eq, FacilityType.WELL_PAD)
        assert ctx.exceeds_reporting_threshold is True


# ── Report Generation ────────────────────────────────────────────────


class TestReportGenerator:
    def test_generate_report(self):
        detection = _make_detection()
        facility = _make_facility()
        chain = ChainOfCustody(description="Test attribution report")
        # Simulate prior chain links
        chain.add_link(
            action=CustodyAction.INGEST,
            actor="test",
            description="Test ingest",
            input_hashes=["abc"],
        )

        gen = ReportGenerator()
        report = gen.generate(detection, facility, chain)

        assert report.report_id.startswith("RPT-")
        assert report.detection_id == "DET-TEST-001"
        assert report.corroboration_level == "triple"
        assert report.facility["facility_type"] == "well_pad"
        assert report.emission["rate_kg_h"] == 500.0
        assert report.regulatory["severity"] == "major"
        assert report.chain_integrity_verified is True
        assert report.chain_link_count == 2  # 1 ingest + 1 report

    def test_report_json_serializable(self):
        detection = _make_detection()
        facility = _make_facility()
        chain = ChainOfCustody(description="JSON test")

        gen = ReportGenerator()
        report = gen.generate(detection, facility, chain)

        j = report.to_json()
        data = json.loads(j)
        assert data["case_id"] == chain.case_id
        assert "detection" in data
        assert "facility" in data
        assert "emission" in data
        assert "regulatory" in data
        assert "chain_of_custody" in data

    def test_report_dict_structure(self):
        detection = _make_detection()
        facility = _make_facility()
        chain = ChainOfCustody(description="Dict test")

        gen = ReportGenerator()
        report = gen.generate(detection, facility, chain)
        d = report.to_dict()

        # Verify nested structure
        assert d["detection"]["sensors_used"] == []  # No obs in this test detection
        assert d["facility"]["operator"] == "Test Oil Co."
        assert d["emission"]["gwp_basis"] == "GWP-20"
        assert d["chain_of_custody"]["integrity_verified"] is True

    def test_generate_multi(self):
        detection = _make_detection()
        facilities = [
            _make_facility(FacilityType.WELL_PAD),
            _make_facility(FacilityType.COMPRESSOR_STATION),
        ]
        chain = ChainOfCustody(description="Multi-report test")

        gen = ReportGenerator()
        reports = gen.generate_multi(detection, facilities, chain)

        assert len(reports) == 2
        assert reports[0].facility["facility_type"] == "well_pad"
        assert reports[1].facility["facility_type"] == "compressor_station"
        # Chain should have 2 report links
        report_links = [l for l in chain.links if l.action == CustodyAction.REPORT]
        assert len(report_links) == 2

    def test_chain_integrity_after_report(self):
        detection = _make_detection()
        facility = _make_facility()
        chain = ChainOfCustody(description="Integrity test")
        chain.add_link(
            action=CustodyAction.FUSE,
            actor="test",
            description="Prior fusion step",
            input_hashes=["x"],
        )

        gen = ReportGenerator()
        report = gen.generate(detection, facility, chain)

        # Chain should still be intact after report generation
        valid, errors = chain.verify_integrity()
        assert valid is True, f"Chain broken: {errors}"
        assert len(chain.links) == 2
