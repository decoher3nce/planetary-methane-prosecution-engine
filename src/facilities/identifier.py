"""
Sentinel-2 Facility Identification Engine.

Analyzes Sentinel-2 10m optical imagery to identify physical infrastructure
near detected methane plumes. Uses spectral indices (NDVI, NDBI) and
spatial pattern analysis to classify facility types and compute
attribution scores.

In production, this would operate on actual Sentinel-2 band data (B2-B12).
The current implementation works with pre-computed spectral indices and
facility footprints, suitable for integration with GEE or local raster
processing pipelines.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from src.evidence.chain_of_custody import ChainOfCustody, CustodyAction
from src.facilities.models import (
    Facility,
    FacilityFootprint,
    FacilityType,
    InfrastructureSignature,
)
from src.fusion.engine import FusedDetection, haversine_distance_m
from src.sensors.models import BoundingBox, Coordinate, Sentinel2Observation


# ── Spectral classification thresholds ───────────────────────────────
# Based on Sentinel-2 band ratios for infrastructure detection

# NDVI < threshold → bare earth / built-up (not vegetation)
NDVI_BARE_THRESHOLD = 0.15
# NDBI > threshold → built-up area
NDBI_BUILT_THRESHOLD = 0.0
# Area thresholds in m² for facility type classification
WELL_PAD_AREA_RANGE = (500, 25_000)  # Typical well pad: 0.05–2.5 hectares
COMPRESSOR_AREA_RANGE = (2_000, 50_000)
PROCESSING_PLANT_AREA_RANGE = (10_000, 500_000)
LANDFILL_AREA_RANGE = (50_000, 5_000_000)
TANK_BATTERY_AREA_RANGE = (200, 10_000)


@dataclass
class SpectralIndices:
    """Sentinel-2 derived spectral indices for a candidate footprint."""

    ndvi: float  # (B8 - B4) / (B8 + B4)  — vegetation
    ndbi: float  # (B11 - B8) / (B11 + B8) — built-up
    bsi: float   # Bare soil index
    ndwi: float  # (B3 - B8) / (B3 + B8)  — water


@dataclass
class CandidateFacility:
    """A candidate facility detected in Sentinel-2 imagery before classification."""

    centroid: Coordinate
    bbox: BoundingBox
    area_m2: float
    perimeter_m: float
    orientation_deg: float
    spectral: SpectralIndices
    signatures: List[InfrastructureSignature]


class FacilityIdentifier:
    """Identifies and classifies facilities near methane detections using Sentinel-2.

    The identification pipeline:
    1. Define search radius around plume center
    2. Analyze Sentinel-2 spectral indices within search area
    3. Detect candidate facility footprints via spectral thresholding
    4. Classify each candidate by type using area + spectral signatures
    5. Compute attribution scores (distance, wind direction, confidence)
    6. Record all steps in chain of custody
    """

    # Search radius around plume center (meters)
    SEARCH_RADIUS_M: float = 2000.0

    # Maximum distance to attribute a facility to a plume
    MAX_ATTRIBUTION_DISTANCE_M: float = 1500.0

    # Minimum confidence to include a facility in results
    MIN_CONFIDENCE: float = 0.3

    def __init__(self) -> None:
        self._sentinel2_observations: List[Sentinel2Observation] = []

    def add_observation(self, obs: Sentinel2Observation) -> None:
        """Register a Sentinel-2 observation for facility search."""
        self._sentinel2_observations.append(obs)

    def identify_facilities(
        self,
        detection: FusedDetection,
        candidates: List[CandidateFacility],
        wind_direction_deg: Optional[float] = None,
        chain: Optional[ChainOfCustody] = None,
    ) -> List[Facility]:
        """Identify and classify facilities near a fused methane detection.

        Args:
            detection: A FusedDetection with plume location.
            candidates: Pre-extracted candidate facility footprints from
                        Sentinel-2 analysis (spectral thresholding + segmentation).
            wind_direction_deg: Wind direction in degrees (where wind comes FROM).
                                Used to determine upwind/downwind attribution.
            chain: Optional chain of custody for evidence tracking.

        Returns:
            List of classified Facility objects with attribution scores.
        """
        facilities: List[Facility] = []
        plume_loc = detection.location

        for candidate in candidates:
            # Distance check
            distance = haversine_distance_m(plume_loc, candidate.centroid)
            if distance > self.MAX_ATTRIBUTION_DISTANCE_M:
                continue

            # Classify facility type
            facility_type, type_confidence = self._classify_type(candidate)

            # Compute wind attribution
            upwind = None
            if wind_direction_deg is not None:
                upwind = self._is_upwind(
                    plume_loc, candidate.centroid, wind_direction_deg
                )

            # Compute overall identification confidence
            confidence = self._compute_confidence(
                candidate, distance, upwind, type_confidence
            )

            if confidence < self.MIN_CONFIDENCE:
                continue

            # Build facility object
            facility = Facility(
                facility_id=f"FAC-{uuid4().hex[:8].upper()}",
                facility_type=facility_type,
                footprint=FacilityFootprint(
                    centroid=candidate.centroid,
                    bounding_box=candidate.bbox,
                    area_m2=candidate.area_m2,
                    perimeter_m=candidate.perimeter_m,
                    orientation_deg=candidate.orientation_deg,
                ),
                signatures_detected=candidate.signatures,
                identification_confidence=confidence,
                ndvi_mean=candidate.spectral.ndvi,
                ndbi_mean=candidate.spectral.ndbi,
                distance_to_plume_m=distance,
                upwind_of_plume=upwind,
            )
            facilities.append(facility)

        # Sort by attribution strength (closest upwind facilities first)
        facilities.sort(
            key=lambda f: (
                not (f.upwind_of_plume or False),  # Upwind first
                f.distance_to_plume_m or float("inf"),  # Then by distance
            )
        )

        # Record in chain of custody
        if chain is not None and facilities:
            chain.add_link(
                action=CustodyAction.ATTRIBUTE,
                actor="pmpe.facilities.identifier",
                description=(
                    f"Identified {len(facilities)} facilities near detection "
                    f"{detection.detection_id} within {self.MAX_ATTRIBUTION_DISTANCE_M}m"
                ),
                input_hashes=[
                    hashlib.sha256(
                        detection.detection_id.encode()
                    ).hexdigest()
                ],
                output_hash=hashlib.sha256(
                    json.dumps(
                        [f.to_attribution_dict() for f in facilities],
                        sort_keys=True,
                    ).encode()
                ).hexdigest(),
                metadata={
                    "detection_id": detection.detection_id,
                    "facilities_found": len(facilities),
                    "facility_types": [f.facility_type.value for f in facilities],
                    "search_radius_m": self.SEARCH_RADIUS_M,
                },
            )

        return facilities

    def _classify_type(
        self, candidate: CandidateFacility
    ) -> Tuple[FacilityType, float]:
        """Classify a candidate facility by type using area and spectral signatures.

        Returns (facility_type, classification_confidence).
        """
        area = candidate.area_m2
        sigs = set(candidate.signatures)
        spectral = candidate.spectral

        # Score each possible type
        scores: Dict[FacilityType, float] = {}

        # Well pad: cleared earth, small-medium area, road access
        if WELL_PAD_AREA_RANGE[0] <= area <= WELL_PAD_AREA_RANGE[1]:
            score = 0.4
            if InfrastructureSignature.CLEARED_PAD in sigs:
                score += 0.25
            if InfrastructureSignature.ROAD_ACCESS in sigs:
                score += 0.1
            if InfrastructureSignature.CIRCULAR_TANK in sigs:
                score += 0.15
            if spectral.ndvi < NDVI_BARE_THRESHOLD:
                score += 0.1
            scores[FacilityType.WELL_PAD] = min(score, 1.0)

        # Compressor station: medium area, thermal anomaly
        if COMPRESSOR_AREA_RANGE[0] <= area <= COMPRESSOR_AREA_RANGE[1]:
            score = 0.3
            if InfrastructureSignature.THERMAL_ANOMALY in sigs:
                score += 0.3
            if spectral.ndbi > NDBI_BUILT_THRESHOLD:
                score += 0.15
            if InfrastructureSignature.CLEARED_PAD in sigs:
                score += 0.1
            scores[FacilityType.COMPRESSOR_STATION] = min(score, 1.0)

        # Processing plant: large area, built-up
        if PROCESSING_PLANT_AREA_RANGE[0] <= area <= PROCESSING_PLANT_AREA_RANGE[1]:
            score = 0.25
            if spectral.ndbi > 0.1:
                score += 0.25
            if InfrastructureSignature.THERMAL_ANOMALY in sigs:
                score += 0.2
            if InfrastructureSignature.CIRCULAR_TANK in sigs:
                score += 0.15
            scores[FacilityType.PROCESSING_PLANT] = min(score, 1.0)

        # Pipeline: linear corridor
        if InfrastructureSignature.LINEAR_CORRIDOR in sigs:
            score = 0.5
            if InfrastructureSignature.VEGETATION_STRESS in sigs:
                score += 0.2
            if InfrastructureSignature.DISTURBED_SOIL in sigs:
                score += 0.15
            scores[FacilityType.PIPELINE] = min(score, 1.0)

        # Landfill: very large, impoundment
        if LANDFILL_AREA_RANGE[0] <= area <= LANDFILL_AREA_RANGE[1]:
            score = 0.3
            if InfrastructureSignature.IMPOUNDMENT in sigs:
                score += 0.3
            if InfrastructureSignature.DISTURBED_SOIL in sigs:
                score += 0.15
            if spectral.ndwi > 0.0:
                score += 0.1
            scores[FacilityType.LANDFILL] = min(score, 1.0)

        # Tank battery: small, circular features
        if TANK_BATTERY_AREA_RANGE[0] <= area <= TANK_BATTERY_AREA_RANGE[1]:
            score = 0.35
            if InfrastructureSignature.CIRCULAR_TANK in sigs:
                score += 0.35
            if spectral.ndbi > NDBI_BUILT_THRESHOLD:
                score += 0.15
            scores[FacilityType.TANK_BATTERY] = min(score, 1.0)

        if not scores:
            return FacilityType.UNKNOWN, 0.3

        best_type = max(scores, key=scores.get)
        return best_type, scores[best_type]

    @staticmethod
    def _is_upwind(
        plume_center: Coordinate,
        facility_center: Coordinate,
        wind_from_deg: float,
    ) -> bool:
        """Determine if the facility is upwind of the plume.

        A facility is upwind if it lies in the direction the wind is
        coming FROM, relative to the plume center. The plume extends
        downwind, so the source should be upwind.

        Args:
            plume_center: Center of the detected plume.
            facility_center: Center of the candidate facility.
            wind_from_deg: Direction wind is coming FROM (0=N, 90=E).

        Returns:
            True if facility is approximately upwind of the plume.
        """
        # Bearing from plume center to facility
        dlon = math.radians(facility_center.lon - plume_center.lon)
        lat1 = math.radians(plume_center.lat)
        lat2 = math.radians(facility_center.lat)

        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        bearing = (math.degrees(math.atan2(x, y)) + 360) % 360

        # Angular difference between bearing-to-facility and wind-from direction
        # If facility is in the direction wind comes from, it's upwind
        diff = abs(bearing - wind_from_deg)
        if diff > 180:
            diff = 360 - diff

        # Within 60° of wind-from direction = upwind
        return diff <= 60

    @staticmethod
    def _compute_confidence(
        candidate: CandidateFacility,
        distance_m: float,
        upwind: Optional[bool],
        type_confidence: float,
    ) -> float:
        """Compute overall facility identification confidence.

        Factors:
        - Type classification confidence (from spectral/area analysis)
        - Proximity to plume (closer = higher confidence)
        - Wind attribution (upwind = higher confidence)
        - Number of corroborating signatures
        """
        # Start with type classification confidence (0-1), weighted 40%
        score = type_confidence * 0.4

        # Proximity score: 1.0 at 0m, 0.0 at MAX_ATTRIBUTION_DISTANCE
        proximity = max(0.0, 1.0 - distance_m / 1500.0)
        score += proximity * 0.25

        # Wind attribution bonus
        if upwind is True:
            score += 0.15
        elif upwind is None:
            score += 0.05  # No wind data — slight default

        # Signature count bonus (more signatures = more certain)
        n_sigs = len(candidate.signatures)
        sig_bonus = min(n_sigs * 0.05, 0.20)
        score += sig_bonus

        return min(max(score, 0.0), 1.0)
