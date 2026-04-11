"""
End-to-end integration test: full prosecution pipeline.

Simulates a complete Permian Basin super-emitter prosecution from
raw sensor observations through sealed court-ready evidence package.

Pipeline:
  1. Ingest observations from 4 sensors
  2. Fuse into corroborated detection
  3. Identify responsible facility via Sentinel-2
  4. Generate attribution report
  5. Package as court-ready evidence
  6. Verify entire chain of custody
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.evidence.chain_of_custody import ChainOfCustody, CustodyAction
from src.evidence.packager import EvidencePackager
from src.facilities.identifier import (
    CandidateFacility,
    FacilityIdentifier,
    SpectralIndices,
)
from src.facilities.models import FacilityType, InfrastructureSignature
from src.fusion.engine import (
    CorroborationLevel,
    FusionConfidence,
    FusionEngine,
)
from src.reports.attribution import ReportGenerator, SeverityLevel
from src.sensors.models import (
    BoundingBox,
    Coordinate,
    GHGSatObservation,
    MethaneSATObservation,
    ProcessingLevel,
    Sentinel5PObservation,
    Tanager1Observation,
)

NOW = datetime(2026, 3, 15, 14, 30, 0, tzinfo=timezone.utc)
PERMIAN_BBOX = BoundingBox(min_lon=-103.8, min_lat=31.7, max_lon=-103.6, max_lat=31.9)
WIDE_BBOX = BoundingBox(min_lon=-104.5, min_lat=31.0, max_lon=-103.0, max_lat=32.5)
PLUME_CENTER = Coordinate(lon=-103.72, lat=31.81)


class TestFullProsecutionPipeline:
    """End-to-end: sensor data → sealed evidence package."""

    def test_permian_basin_super_emitter(self):
        # ── Initialize chain of custody ──────────────────────────
        chain = ChainOfCustody(
            description="Permian Basin super-emitter — Well Pad WP-4117"
        )

        # ── Step 1: Ingest sensor observations ──────────────────
        engine = FusionEngine()

        ghgsat = GHGSatObservation(
            observation_id="GHG-2026-0315-PB001",
            acquisition_time=NOW,
            processing_level=ProcessingLevel.L2,
            bounding_box=PERMIAN_BBOX,
            data_uri="s3://ghgsat/permian/2026-03-15/obs_001.nc",
            quality_flag=0.95,
            plume_detected=True,
            plume_center=PLUME_CENTER,
            emission_rate_kg_h=500.0,
            emission_rate_uncertainty_kg_h=75.0,
        )

        tanager = Tanager1Observation(
            observation_id="TAN-2026-0315-PB001",
            acquisition_time=NOW + timedelta(hours=2),
            processing_level=ProcessingLevel.L2,
            bounding_box=PERMIAN_BBOX,
            data_uri="s3://tanager/permian/2026-03-15/scene_001.nc",
            quality_flag=0.92,
            methane_band_depth_1650nm=0.045,
            methane_band_depth_2300nm=0.032,
            swir_snr=250.0,
        )

        methanesat = MethaneSATObservation(
            observation_id="MSAT-2026-0315-PB001",
            acquisition_time=NOW + timedelta(hours=6),
            processing_level=ProcessingLevel.L2,
            bounding_box=WIDE_BBOX,
            data_uri="s3://methanesat/permian/2026-03-15/swath_001.nc",
            quality_flag=0.88,
            area_flux_kg_km2_h=12.5,
            enhancement_ppb=45.0,
            background_ch4_ppb=1900.0,
        )

        sentinel5p = Sentinel5PObservation(
            observation_id="S5P-2026-0310-PB001",
            acquisition_time=NOW - timedelta(days=5),
            processing_level=ProcessingLevel.L2,
            bounding_box=WIDE_BBOX,
            data_uri="s3://sentinel5p/CH4/2026-03-10/orbit_12345.nc",
            quality_flag=0.7,
            xch4_ppb=1950.0,
            qa_value=0.75,
            cloud_fraction=0.05,
        )

        # Ingest all with chain tracking
        for obs in [ghgsat, tanager, methanesat, sentinel5p]:
            engine.ingest(obs, chain=chain, raw_bytes=f"raw-{obs.observation_id}".encode())

        assert len(chain.links) == 4

        # ── Step 2: Fuse into corroborated detection ────────────
        detection = engine.fuse(ghgsat, chain=chain)

        assert detection.corroboration_level in (
            CorroborationLevel.TRIPLE,
            CorroborationLevel.QUAD,
        )
        assert detection.spectral_confirmed is True
        assert detection.emission_rate_kg_h == 500.0
        assert detection.area_flux_kg_km2_h == 12.5
        assert detection.sensor_count() >= 3

        # ── Step 3: Identify facility via Sentinel-2 ────────────
        identifier = FacilityIdentifier()

        # Simulate Sentinel-2 derived candidates
        # Well pad SW of plume — upwind when wind from SW (225°)
        well_pad = CandidateFacility(
            centroid=Coordinate(lon=-103.723, lat=31.808),
            bbox=BoundingBox(
                min_lon=-103.724, min_lat=31.807,
                max_lon=-103.722, max_lat=31.809,
            ),
            area_m2=5000,
            perimeter_m=300,
            orientation_deg=45.0,
            spectral=SpectralIndices(ndvi=0.08, ndbi=0.12, bsi=0.3, ndwi=-0.2),
            signatures=[
                InfrastructureSignature.CLEARED_PAD,
                InfrastructureSignature.ROAD_ACCESS,
                InfrastructureSignature.CIRCULAR_TANK,
            ],
        )

        compressor = CandidateFacility(
            centroid=Coordinate(lon=-103.71, lat=31.815),
            bbox=BoundingBox(
                min_lon=-103.712, min_lat=31.813,
                max_lon=-103.708, max_lat=31.817,
            ),
            area_m2=12000,
            perimeter_m=450,
            orientation_deg=90.0,
            spectral=SpectralIndices(ndvi=0.05, ndbi=0.2, bsi=0.25, ndwi=-0.15),
            signatures=[
                InfrastructureSignature.THERMAL_ANOMALY,
                InfrastructureSignature.CLEARED_PAD,
            ],
        )

        facilities = identifier.identify_facilities(
            detection,
            candidates=[well_pad, compressor],
            wind_direction_deg=225.0,  # Wind from SW
            chain=chain,
        )

        assert len(facilities) >= 1
        # Well pad should be primary attribution (closest, upwind)
        primary = facilities[0]
        assert primary.facility_type == FacilityType.WELL_PAD
        assert primary.upwind_of_plume is True

        # ── Step 4: Generate attribution report ─────────────────
        gen = ReportGenerator()
        report = gen.generate(detection, primary, chain)

        assert report.corroboration_level == detection.corroboration_level.value
        assert report.emission["rate_kg_h"] == 500.0
        assert report.emission["rate_co2e_tonnes_year"] > 0
        assert report.regulatory["severity"] == SeverityLevel.MAJOR.value
        assert report.regulatory["qualifies_as_super_emitter"] is True

        # ── Step 5: Package as court-ready evidence ─────────────
        packager = EvidencePackager()
        package = packager.package(
            report, chain,
            additional_artifacts={
                "plume_visualization.png": b"<synthetic plume image>",
            },
        )

        assert len(package.artifacts) >= 5
        assert package.seal is not None
        assert package.verify_seal() is True

        # Verify all artifacts intact
        errors = package.verify_artifacts()
        assert errors == [], f"Artifact verification failed: {errors}"

        # ── Step 6: Verify entire chain of custody ──────────────
        valid, chain_errors = chain.verify_integrity()
        assert valid is True, f"Chain integrity failed: {chain_errors}"

        # Chain should have: 4 ingests + 1 fuse + 1 attribute + 1 report + 1 package
        assert len(chain.links) == 8

        actions = [link.action for link in chain.links]
        assert actions.count(CustodyAction.INGEST) == 4
        assert actions.count(CustodyAction.FUSE) == 1
        assert actions.count(CustodyAction.ATTRIBUTE) == 1
        assert actions.count(CustodyAction.REPORT) == 1
        assert actions.count(CustodyAction.PACKAGE) == 1

        # ── Verify rendered report content ──────────────────────
        rendered = package.artifacts["attribution_report.txt"].decode("utf-8")
        assert "PLANETARY METHANE PROSECUTION ENGINE" in rendered
        assert "DET-GHG-2026-0315-PB001" in rendered
        assert "Well Pad" in rendered
        assert "500" in rendered
        assert "VERIFIED" in rendered
        assert "Test Oil" not in rendered  # No operator set in this test

        # ── Verify evidence bundle ──────────────────────────────
        bundle = json.loads(package.artifacts["evidence_bundle.json"])
        assert bundle["detection"]["corroboration_level"] in ("triple", "quad")
        assert bundle["regulatory"]["qualifies_as_super_emitter"] is True
        assert bundle["chain_of_custody"]["integrity_verified"] is True

        # ── Summary ─────────────────────────────────────────────
        summary = package.to_summary()
        assert summary["seal_valid"] is True
