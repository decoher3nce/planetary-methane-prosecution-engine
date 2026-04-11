"""Tests for the multi-sensor fusion engine."""

from datetime import datetime, timedelta, timezone

import pytest

from src.evidence.chain_of_custody import ChainOfCustody, CustodyAction
from src.fusion.engine import (
    CorroborationLevel,
    FusedDetection,
    FusionConfidence,
    FusionEngine,
    bbox_center,
    bboxes_overlap,
    haversine_distance_m,
)
from src.sensors.models import (
    BoundingBox,
    Coordinate,
    GHGSatObservation,
    MethaneSATObservation,
    ProcessingLevel,
    Sentinel5PObservation,
    Tanager1Observation,
)

# ── Test fixtures: Permian Basin super-emitter scenario ──────────────

NOW = datetime(2026, 3, 15, 14, 30, 0, tzinfo=timezone.utc)
PERMIAN_BBOX = BoundingBox(min_lon=-103.8, min_lat=31.7, max_lon=-103.6, max_lat=31.9)
PLUME_CENTER = Coordinate(lon=-103.72, lat=31.81)


def make_ghgsat(
    plume: bool = True,
    emission_rate: float = 500.0,
    time_offset: timedelta = timedelta(0),
) -> GHGSatObservation:
    return GHGSatObservation(
        observation_id="GHG-TEST-001",
        acquisition_time=NOW + time_offset,
        processing_level=ProcessingLevel.L2,
        bounding_box=PERMIAN_BBOX,
        data_uri="s3://ghgsat/test.nc",
        quality_flag=0.95,
        plume_detected=plume,
        plume_center=PLUME_CENTER if plume else None,
        emission_rate_kg_h=emission_rate,
        emission_rate_uncertainty_kg_h=75.0,
    )


def make_tanager(
    band_1650: float = 0.045,
    band_2300: float = 0.032,
    time_offset: timedelta = timedelta(hours=2),
    bbox: BoundingBox = None,
) -> Tanager1Observation:
    return Tanager1Observation(
        observation_id="TAN-TEST-001",
        acquisition_time=NOW + time_offset,
        processing_level=ProcessingLevel.L2,
        bounding_box=bbox or PERMIAN_BBOX,
        data_uri="s3://tanager/test.nc",
        quality_flag=0.92,
        methane_band_depth_1650nm=band_1650,
        methane_band_depth_2300nm=band_2300,
        swir_snr=250.0,
    )


def make_methanesat(
    time_offset: timedelta = timedelta(hours=6),
) -> MethaneSATObservation:
    # Wide bbox to encompass the plume center
    wide_bbox = BoundingBox(min_lon=-104.5, min_lat=31.0, max_lon=-103.0, max_lat=32.5)
    return MethaneSATObservation(
        observation_id="MSAT-TEST-001",
        acquisition_time=NOW + time_offset,
        processing_level=ProcessingLevel.L2,
        bounding_box=wide_bbox,
        data_uri="s3://methanesat/test.nc",
        quality_flag=0.88,
        area_flux_kg_km2_h=12.5,
        enhancement_ppb=45.0,
        background_ch4_ppb=1900.0,
    )


def make_sentinel5p(
    xch4: float = 1950.0,
    time_offset: timedelta = timedelta(days=-5),
) -> Sentinel5PObservation:
    wide_bbox = BoundingBox(min_lon=-104.5, min_lat=31.0, max_lon=-103.0, max_lat=32.5)
    return Sentinel5PObservation(
        observation_id=f"S5P-TEST-{abs(time_offset.days):03d}",
        acquisition_time=NOW + time_offset,
        processing_level=ProcessingLevel.L2,
        bounding_box=wide_bbox,
        data_uri=f"s3://sentinel5p/test_{abs(time_offset.days)}.nc",
        quality_flag=0.7,
        xch4_ppb=xch4,
        qa_value=0.75,
        cloud_fraction=0.05,
    )


# ── Utility function tests ──────────────────────────────────────────


class TestHaversine:
    def test_same_point(self):
        c = Coordinate(lon=-103.72, lat=31.81)
        assert haversine_distance_m(c, c) == pytest.approx(0.0, abs=0.1)

    def test_known_distance(self):
        # ~111 km per degree of latitude
        a = Coordinate(lon=0.0, lat=0.0)
        b = Coordinate(lon=0.0, lat=1.0)
        d = haversine_distance_m(a, b)
        assert 110_000 < d < 112_000

    def test_nearby_points(self):
        a = Coordinate(lon=-103.72, lat=31.81)
        b = Coordinate(lon=-103.721, lat=31.811)
        d = haversine_distance_m(a, b)
        assert d < 500  # Should be well within tolerance


class TestBboxUtils:
    def test_bbox_center(self):
        bb = BoundingBox(min_lon=-104.0, min_lat=31.0, max_lon=-103.0, max_lat=32.0)
        c = bbox_center(bb)
        assert c.lon == pytest.approx(-103.5)
        assert c.lat == pytest.approx(31.5)

    def test_bboxes_overlap_true(self):
        a = BoundingBox(min_lon=-104, min_lat=31, max_lon=-103, max_lat=32)
        b = BoundingBox(min_lon=-103.5, min_lat=31.5, max_lon=-102.5, max_lat=32.5)
        assert bboxes_overlap(a, b) is True

    def test_bboxes_overlap_false(self):
        a = BoundingBox(min_lon=-104, min_lat=31, max_lon=-103, max_lat=32)
        b = BoundingBox(min_lon=-100, min_lat=35, max_lon=-99, max_lat=36)
        assert bboxes_overlap(a, b) is False


# ── Fusion engine tests ─────────────────────────────────────────────


class TestFusionEngineIngest:
    def test_ingest_valid(self):
        engine = FusionEngine()
        obs = make_ghgsat()
        engine.ingest(obs)

    def test_ingest_with_chain(self):
        engine = FusionEngine()
        chain = ChainOfCustody(description="Test")
        obs = make_ghgsat()
        engine.ingest(obs, chain=chain, raw_bytes=b"test data")
        assert len(chain.links) == 1
        assert chain.links[0].action == CustodyAction.INGEST
        assert obs.content_hash is not None

    def test_ingest_low_quality_rejected(self):
        engine = FusionEngine()
        obs = make_ghgsat()
        obs.quality_flag = 0.2
        with pytest.raises(ValueError, match="quality"):
            engine.ingest(obs)


class TestFusionEngineFuse:
    def test_single_sensor_detection(self):
        """GHGSat only — single corroboration."""
        engine = FusionEngine()
        ghg = make_ghgsat()
        engine.ingest(ghg)
        det = engine.fuse(ghg)

        assert det.corroboration_level == CorroborationLevel.SINGLE
        assert det.emission_rate_kg_h == 500.0
        assert det.sensor_count() == 1

    def test_dual_sensor_ghgsat_tanager(self):
        """GHGSat + Tanager-1 — spectral confirmation."""
        engine = FusionEngine()
        ghg = make_ghgsat()
        tan = make_tanager()
        engine.ingest(ghg)
        engine.ingest(tan)
        det = engine.fuse(ghg)

        assert det.corroboration_level == CorroborationLevel.DUAL
        assert det.spectral_confirmed is True
        assert det.ch4_band_depth_1650nm == 0.045
        assert det.sensor_count() == 2

    def test_triple_sensor_prosecution_grade(self):
        """GHGSat + Tanager-1 + MethaneSAT — prosecution grade."""
        engine = FusionEngine()
        ghg = make_ghgsat()
        tan = make_tanager()
        msat = make_methanesat()
        for obs in [ghg, tan, msat]:
            engine.ingest(obs)

        det = engine.fuse(ghg)
        assert det.corroboration_level == CorroborationLevel.TRIPLE
        assert det.confidence == FusionConfidence.HIGH
        assert det.area_flux_kg_km2_h == 12.5
        assert det.sensor_count() == 3

    def test_quad_sensor_maximum_corroboration(self):
        """All 4 sensors — maximum corroboration."""
        engine = FusionEngine()
        ghg = make_ghgsat()
        tan = make_tanager()
        msat = make_methanesat()
        # Multiple S5P observations for persistence
        s5p_1 = make_sentinel5p(time_offset=timedelta(days=-15))
        s5p_2 = make_sentinel5p(time_offset=timedelta(days=-7))
        s5p_3 = make_sentinel5p(time_offset=timedelta(days=-1))

        for obs in [ghg, tan, msat, s5p_1, s5p_2, s5p_3]:
            engine.ingest(obs)

        det = engine.fuse(ghg)
        assert det.corroboration_level == CorroborationLevel.QUAD
        assert det.confidence == FusionConfidence.VERY_HIGH
        assert det.sentinel5p_corroborated is True
        assert det.temporal_persistence_days >= 14
        assert det.sensor_count() == 4

    def test_no_plume_raises(self):
        engine = FusionEngine()
        ghg = make_ghgsat(plume=False)
        engine.ingest(ghg)
        with pytest.raises(ValueError, match="no plume"):
            engine.fuse(ghg)

    def test_tanager_outside_spatial_tolerance(self):
        """Tanager-1 far away should not match."""
        engine = FusionEngine()
        ghg = make_ghgsat()
        far_bbox = BoundingBox(min_lon=-100.0, min_lat=35.0, max_lon=-99.8, max_lat=35.2)
        tan = make_tanager(bbox=far_bbox)
        engine.ingest(ghg)
        engine.ingest(tan)

        det = engine.fuse(ghg)
        assert det.sensor_count() == 1  # Only GHGSat
        assert det.spectral_confirmed is False

    def test_tanager_outside_temporal_tolerance(self):
        """Tanager-1 too old should not match."""
        engine = FusionEngine()
        ghg = make_ghgsat()
        tan = make_tanager(time_offset=timedelta(days=10))
        engine.ingest(ghg)
        engine.ingest(tan)

        det = engine.fuse(ghg)
        assert det.sensor_count() == 1

    def test_tanager_weak_spectral_no_confirm(self):
        """Tanager-1 with weak CH4 bands should not confirm."""
        engine = FusionEngine()
        ghg = make_ghgsat()
        tan = make_tanager(band_1650=0.005, band_2300=0.003)  # Below threshold
        engine.ingest(ghg)
        engine.ingest(tan)

        det = engine.fuse(ghg)
        assert det.spectral_confirmed is False
        # Observation not added since spectral check failed
        assert det.sensor_count() == 1


class TestFusionWithChainOfCustody:
    def test_full_chain(self):
        """End-to-end: ingest + fuse with complete chain of custody."""
        engine = FusionEngine()
        chain = ChainOfCustody(description="Permian Basin super-emitter prosecution")

        ghg = make_ghgsat()
        tan = make_tanager()
        msat = make_methanesat()

        engine.ingest(ghg, chain=chain, raw_bytes=b"ghgsat-raw")
        engine.ingest(tan, chain=chain, raw_bytes=b"tanager-raw")
        engine.ingest(msat, chain=chain, raw_bytes=b"methanesat-raw")

        det = engine.fuse(ghg, chain=chain)

        # Chain should have 4 links: 3 ingests + 1 fuse
        assert len(chain.links) == 4
        assert chain.links[0].action == CustodyAction.INGEST
        assert chain.links[3].action == CustodyAction.FUSE

        # Chain integrity should hold
        valid, errors = chain.verify_integrity()
        assert valid is True, f"Chain integrity failed: {errors}"

        # Detection should reference the chain
        assert det.chain_of_custody is chain
        summary = det.to_evidence_summary()
        assert summary["chain_of_custody_case_id"] == chain.case_id


class TestFusedDetection:
    def test_evidence_summary(self):
        det = FusedDetection(
            detection_id="DET-TEST",
            location=PLUME_CENTER,
            detection_time=NOW,
            observations=[make_ghgsat()],
            corroboration_level=CorroborationLevel.SINGLE,
            confidence=FusionConfidence.LOW,
            confidence_score=0.35,
            emission_rate_kg_h=500.0,
        )
        summary = det.to_evidence_summary()
        assert summary["detection_id"] == "DET-TEST"
        assert summary["emission_rate_kg_h"] == 500.0
        assert summary["confidence"] == "low"
