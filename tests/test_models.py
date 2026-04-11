"""Tests for sensor data models."""

from datetime import datetime, timezone

import pytest

from src.sensors.models import (
    BoundingBox,
    Coordinate,
    GHGSatObservation,
    MethaneSATObservation,
    ProcessingLevel,
    Sentinel2Observation,
    Sentinel5PObservation,
    SensorType,
    Tanager1Observation,
)


PERMIAN_BBOX = BoundingBox(min_lon=-104.5, min_lat=31.0, max_lon=-103.0, max_lat=32.5)
NOW = datetime.now(timezone.utc)


class TestTanager1:
    def test_create_observation(self):
        obs = Tanager1Observation(
            observation_id="TAN-2026-001",
            acquisition_time=NOW,
            processing_level=ProcessingLevel.L2,
            bounding_box=PERMIAN_BBOX,
            data_uri="s3://tanager/scene_001.nc",
            quality_flag=0.92,
            methane_band_depth_1650nm=0.045,
            methane_band_depth_2300nm=0.032,
            swir_snr=250.0,
        )
        assert obs.sensor == SensorType.TANAGER_1
        assert obs.num_bands == 400
        assert obs.spatial_resolution_m == 30.0

    def test_content_hash(self):
        obs = Tanager1Observation(
            observation_id="TAN-2026-002",
            acquisition_time=NOW,
            processing_level=ProcessingLevel.L1B,
            bounding_box=PERMIAN_BBOX,
            data_uri="s3://tanager/scene_002.nc",
            quality_flag=0.85,
        )
        h = obs.compute_content_hash(b"fake raw hyperspectral data")
        assert len(h) == 64  # SHA-256 hex
        assert obs.content_hash == h

    def test_evidence_dict(self):
        obs = Tanager1Observation(
            observation_id="TAN-2026-003",
            acquisition_time=NOW,
            processing_level=ProcessingLevel.L2,
            bounding_box=PERMIAN_BBOX,
            data_uri="s3://tanager/scene_003.nc",
            quality_flag=0.9,
        )
        d = obs.to_evidence_dict()
        assert d["sensor"] == "planet_tanager_1"
        assert "acquisition_time" in d


class TestGHGSat:
    def test_plume_detection(self):
        obs = GHGSatObservation(
            observation_id="GHG-2026-001",
            acquisition_time=NOW,
            processing_level=ProcessingLevel.L2,
            bounding_box=PERMIAN_BBOX,
            data_uri="s3://ghgsat/obs_001.nc",
            quality_flag=0.95,
            plume_detected=True,
            plume_center=Coordinate(lon=-103.7, lat=31.8),
            emission_rate_kg_h=500.0,
            emission_rate_uncertainty_kg_h=75.0,
        )
        assert obs.plume_detected is True
        assert obs.emission_rate_kg_h == 500.0
        assert obs.spatial_resolution_m == 25.0


class TestMethaneSAT:
    def test_area_flux(self):
        obs = MethaneSATObservation(
            observation_id="MSAT-2026-001",
            acquisition_time=NOW,
            processing_level=ProcessingLevel.L2,
            bounding_box=PERMIAN_BBOX,
            data_uri="s3://methanesat/obs_001.nc",
            quality_flag=0.88,
            area_flux_kg_km2_h=12.5,
            enhancement_ppb=45.0,
            background_ch4_ppb=1900.0,
        )
        assert obs.swath_width_km == 200.0
        assert obs.area_flux_kg_km2_h == 12.5


class TestSentinel5P:
    def test_tropomi(self):
        obs = Sentinel5PObservation(
            observation_id="S5P-2026-001",
            acquisition_time=NOW,
            processing_level=ProcessingLevel.L2,
            bounding_box=PERMIAN_BBOX,
            data_uri="s3://sentinel5p/CH4_001.nc",
            quality_flag=0.7,
            xch4_ppb=1950.0,
            qa_value=0.75,
            cloud_fraction=0.05,
        )
        assert obs.sensor == SensorType.SENTINEL_5P
        assert obs.xch4_ppb == 1950.0


class TestSentinel2:
    def test_optical(self):
        obs = Sentinel2Observation(
            observation_id="S2A-2026-001",
            acquisition_time=NOW,
            processing_level=ProcessingLevel.L2,
            bounding_box=PERMIAN_BBOX,
            data_uri="s3://sentinel2/T13SPA.zip",
            quality_flag=0.98,
            cloud_cover_pct=2.5,
            tile_id="T13SPA",
        )
        assert obs.spatial_resolution_m == 10.0
        assert obs.cloud_cover_pct == 2.5


class TestBoundingBox:
    def test_valid_bbox(self):
        bb = BoundingBox(min_lon=-104.5, min_lat=31.0, max_lon=-103.0, max_lat=32.5)
        assert bb.min_lon == -104.5

    def test_invalid_lon_order(self):
        with pytest.raises(ValueError):
            BoundingBox(min_lon=-103.0, min_lat=31.0, max_lon=-104.5, max_lat=32.5)

    def test_invalid_lat_order(self):
        with pytest.raises(ValueError):
            BoundingBox(min_lon=-104.5, min_lat=32.5, max_lon=-103.0, max_lat=31.0)
