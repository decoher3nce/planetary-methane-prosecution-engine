"""Tests for Sentinel-2 facility identification and attribution."""

from datetime import datetime, timedelta, timezone

import pytest

from src.evidence.chain_of_custody import ChainOfCustody, CustodyAction
from src.facilities.identifier import (
    CandidateFacility,
    FacilityIdentifier,
    SpectralIndices,
)
from src.facilities.models import (
    Facility,
    FacilityType,
    InfrastructureSignature,
)
from src.fusion.engine import (
    CorroborationLevel,
    FusedDetection,
    FusionConfidence,
)
from src.sensors.models import BoundingBox, Coordinate

NOW = datetime(2026, 3, 15, 14, 30, 0, tzinfo=timezone.utc)
PLUME_CENTER = Coordinate(lon=-103.72, lat=31.81)


def _make_detection() -> FusedDetection:
    return FusedDetection(
        detection_id="DET-TEST-001",
        location=PLUME_CENTER,
        detection_time=NOW,
        corroboration_level=CorroborationLevel.TRIPLE,
        confidence=FusionConfidence.HIGH,
        confidence_score=0.85,
        emission_rate_kg_h=500.0,
    )


def _make_well_pad(
    lon_offset: float = -0.002,
    lat_offset: float = 0.001,
) -> CandidateFacility:
    """Create a candidate that looks like a well pad near the plume."""
    center = Coordinate(
        lon=PLUME_CENTER.lon + lon_offset,
        lat=PLUME_CENTER.lat + lat_offset,
    )
    return CandidateFacility(
        centroid=center,
        bbox=BoundingBox(
            min_lon=center.lon - 0.0003,
            min_lat=center.lat - 0.0002,
            max_lon=center.lon + 0.0003,
            max_lat=center.lat + 0.0002,
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


def _make_compressor(
    lon_offset: float = 0.005,
    lat_offset: float = 0.003,
) -> CandidateFacility:
    """Create a candidate that looks like a compressor station."""
    center = Coordinate(
        lon=PLUME_CENTER.lon + lon_offset,
        lat=PLUME_CENTER.lat + lat_offset,
    )
    return CandidateFacility(
        centroid=center,
        bbox=BoundingBox(
            min_lon=center.lon - 0.001,
            min_lat=center.lat - 0.0008,
            max_lon=center.lon + 0.001,
            max_lat=center.lat + 0.0008,
        ),
        area_m2=15000,
        perimeter_m=500,
        orientation_deg=90.0,
        spectral=SpectralIndices(ndvi=0.05, ndbi=0.2, bsi=0.25, ndwi=-0.15),
        signatures=[
            InfrastructureSignature.THERMAL_ANOMALY,
            InfrastructureSignature.CLEARED_PAD,
        ],
    )


def _make_pipeline() -> CandidateFacility:
    """Create a candidate that looks like a pipeline corridor."""
    center = Coordinate(lon=-103.715, lat=31.812)
    return CandidateFacility(
        centroid=center,
        bbox=BoundingBox(
            min_lon=center.lon - 0.005,
            min_lat=center.lat - 0.0001,
            max_lon=center.lon + 0.005,
            max_lat=center.lat + 0.0001,
        ),
        area_m2=3000,
        perimeter_m=1200,
        orientation_deg=80.0,
        spectral=SpectralIndices(ndvi=0.10, ndbi=0.05, bsi=0.2, ndwi=-0.1),
        signatures=[
            InfrastructureSignature.LINEAR_CORRIDOR,
            InfrastructureSignature.VEGETATION_STRESS,
            InfrastructureSignature.DISTURBED_SOIL,
        ],
    )


def _make_far_facility() -> CandidateFacility:
    """Create a candidate far from the plume — should be excluded."""
    center = Coordinate(lon=-103.5, lat=31.95)  # ~20km away
    return CandidateFacility(
        centroid=center,
        bbox=BoundingBox(
            min_lon=center.lon - 0.001,
            min_lat=center.lat - 0.001,
            max_lon=center.lon + 0.001,
            max_lat=center.lat + 0.001,
        ),
        area_m2=8000,
        perimeter_m=400,
        orientation_deg=0.0,
        spectral=SpectralIndices(ndvi=0.07, ndbi=0.15, bsi=0.28, ndwi=-0.18),
        signatures=[InfrastructureSignature.CLEARED_PAD],
    )


class TestFacilityClassification:
    def test_well_pad_classified(self):
        identifier = FacilityIdentifier()
        detection = _make_detection()
        candidates = [_make_well_pad()]
        facilities = identifier.identify_facilities(detection, candidates)

        assert len(facilities) == 1
        assert facilities[0].facility_type == FacilityType.WELL_PAD

    def test_compressor_classified(self):
        identifier = FacilityIdentifier()
        detection = _make_detection()
        candidates = [_make_compressor()]
        facilities = identifier.identify_facilities(detection, candidates)

        assert len(facilities) == 1
        assert facilities[0].facility_type == FacilityType.COMPRESSOR_STATION

    def test_pipeline_classified(self):
        identifier = FacilityIdentifier()
        detection = _make_detection()
        candidates = [_make_pipeline()]
        facilities = identifier.identify_facilities(detection, candidates)

        assert len(facilities) == 1
        assert facilities[0].facility_type == FacilityType.PIPELINE

    def test_far_facility_excluded(self):
        identifier = FacilityIdentifier()
        detection = _make_detection()
        candidates = [_make_far_facility()]
        facilities = identifier.identify_facilities(detection, candidates)

        assert len(facilities) == 0


class TestWindAttribution:
    def test_upwind_facility_flagged(self):
        identifier = FacilityIdentifier()
        detection = _make_detection()
        # Well pad to the SW of plume — wind from SW means it's upwind
        candidates = [_make_well_pad(lon_offset=-0.002, lat_offset=-0.001)]
        facilities = identifier.identify_facilities(
            detection, candidates, wind_direction_deg=225.0  # Wind from SW
        )

        assert len(facilities) == 1
        assert facilities[0].upwind_of_plume is True

    def test_downwind_facility_flagged(self):
        identifier = FacilityIdentifier()
        detection = _make_detection()
        # Well pad to the NE of plume — wind from SW means it's downwind
        candidates = [_make_well_pad(lon_offset=0.002, lat_offset=0.001)]
        facilities = identifier.identify_facilities(
            detection, candidates, wind_direction_deg=225.0
        )

        assert len(facilities) == 1
        assert facilities[0].upwind_of_plume is False


class TestMultipleFacilities:
    def test_multiple_candidates_ranked(self):
        """Multiple facilities should be sorted: upwind first, then by distance."""
        identifier = FacilityIdentifier()
        detection = _make_detection()
        candidates = [
            _make_well_pad(lon_offset=-0.002, lat_offset=-0.001),  # Close, SW
            _make_compressor(lon_offset=0.005, lat_offset=0.003),  # Farther, NE
            _make_pipeline(),  # Nearby
        ]
        facilities = identifier.identify_facilities(
            detection, candidates, wind_direction_deg=225.0
        )

        assert len(facilities) >= 2
        # First facility should be the upwind one
        assert facilities[0].upwind_of_plume is True

    def test_mixed_with_far_excluded(self):
        identifier = FacilityIdentifier()
        detection = _make_detection()
        candidates = [
            _make_well_pad(),
            _make_far_facility(),  # Should be excluded
            _make_pipeline(),
        ]
        facilities = identifier.identify_facilities(detection, candidates)

        # Far facility should be excluded
        assert len(facilities) == 2
        types = {f.facility_type for f in facilities}
        assert FacilityType.WELL_PAD in types
        assert FacilityType.PIPELINE in types


class TestConfidenceScoring:
    def test_close_upwind_high_confidence(self):
        identifier = FacilityIdentifier()
        detection = _make_detection()
        candidates = [_make_well_pad(lon_offset=-0.001, lat_offset=0.0)]
        facilities = identifier.identify_facilities(
            detection, candidates, wind_direction_deg=270.0  # Wind from W
        )

        assert len(facilities) == 1
        # Close + upwind + good signatures → high confidence
        assert facilities[0].identification_confidence > 0.6

    def test_far_downwind_lower_confidence(self):
        identifier = FacilityIdentifier()
        detection = _make_detection()
        # Facility at edge of attribution distance, downwind
        candidates = [_make_well_pad(lon_offset=0.01, lat_offset=0.005)]
        facilities = identifier.identify_facilities(
            detection, candidates, wind_direction_deg=225.0
        )

        if facilities:  # May be excluded entirely
            assert facilities[0].identification_confidence < 0.7


class TestFacilityChainOfCustody:
    def test_chain_recorded(self):
        identifier = FacilityIdentifier()
        detection = _make_detection()
        chain = ChainOfCustody(description="Facility identification test")
        candidates = [_make_well_pad(), _make_pipeline()]

        facilities = identifier.identify_facilities(
            detection, candidates, chain=chain
        )

        assert len(chain.links) == 1
        assert chain.links[0].action == CustodyAction.ATTRIBUTE
        assert chain.links[0].metadata["facilities_found"] == len(facilities)
        valid, errors = chain.verify_integrity()
        assert valid is True


class TestFacilityModel:
    def test_attribution_dict(self):
        from src.facilities.models import FacilityFootprint

        facility = Facility(
            facility_id="FAC-TEST001",
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
            ndvi_mean=0.08,
            ndbi_mean=0.12,
            distance_to_plume_m=250.0,
            upwind_of_plume=True,
            epa_ghgrp_id="1234567",
            api_well_number="42-461-12345-00",
        )

        d = facility.to_attribution_dict()
        assert d["facility_type"] == "well_pad"
        assert d["operator"] == "Test Oil Co."
        assert d["distance_to_plume_m"] == 250.0
        assert d["regulatory_ids"]["api_well"] == "42-461-12345-00"
