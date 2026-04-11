"""
Multi-Sensor Methane Fusion Engine.

Fuses observations from Tanager-1, GHGSat, MethaneSAT, and Sentinel-5P
into a single, corroborated methane detection with quantified emissions
and a complete chain of custody.

The fusion logic implements a hierarchy of evidence:
  1. GHGSat provides the point-source location and emission rate (25m)
  2. Tanager-1 provides spectral confirmation via CH4 absorption bands (30m)
  3. MethaneSAT provides area-wide context and total basin flux (100m)
  4. Sentinel-5P provides temporal persistence baseline (daily, ~5.5km)

A detection is considered "prosecution-grade" when at least 3 independent
sensors corroborate within spatial and temporal tolerances.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

from src.evidence.chain_of_custody import ChainOfCustody, CustodyAction
from src.sensors.models import (
    BoundingBox,
    Coordinate,
    GHGSatObservation,
    MethaneSATObservation,
    Sentinel5PObservation,
    SensorObservation,
    SensorType,
    Tanager1Observation,
)


class CorroborationLevel(str, Enum):
    """Level of multi-sensor corroboration for a detection."""

    SINGLE = "single"  # 1 sensor — indicative only
    DUAL = "dual"  # 2 sensors — strong evidence
    TRIPLE = "triple"  # 3 sensors — prosecution-grade
    QUAD = "quad"  # 4 sensors — maximum corroboration


class FusionConfidence(str, Enum):
    """Overall confidence in the fused detection."""

    LOW = "low"  # <50% — insufficient for legal use
    MEDIUM = "medium"  # 50-75% — suitable for regulatory notice
    HIGH = "high"  # 75-90% — suitable for enforcement action
    VERY_HIGH = "very_high"  # >90% — suitable for prosecution


@dataclass
class SpatialMatch:
    """Result of spatial co-registration between two observations."""

    obs_a_id: str
    obs_b_id: str
    distance_m: float
    within_tolerance: bool


@dataclass
class FusedDetection:
    """A multi-sensor corroborated methane detection.

    This is the core output of the fusion engine — a single detection
    backed by multiple independent sensor observations with quantified
    confidence and a complete evidence chain.
    """

    detection_id: str
    location: Coordinate
    detection_time: datetime
    observations: List[SensorObservation] = field(default_factory=list)
    corroboration_level: CorroborationLevel = CorroborationLevel.SINGLE
    confidence: FusionConfidence = FusionConfidence.LOW
    confidence_score: float = 0.0

    # Quantified emissions
    emission_rate_kg_h: Optional[float] = None
    emission_rate_uncertainty_kg_h: Optional[float] = None
    emission_rate_source: Optional[str] = None  # Which sensor provided the rate

    # Spectral confirmation
    spectral_confirmed: bool = False
    ch4_band_depth_1650nm: Optional[float] = None
    ch4_band_depth_2300nm: Optional[float] = None

    # Area context
    area_flux_kg_km2_h: Optional[float] = None
    background_ch4_ppb: Optional[float] = None
    enhancement_above_background_ppb: Optional[float] = None

    # Temporal persistence
    temporal_persistence_days: Optional[int] = None
    sentinel5p_corroborated: bool = False

    # Spatial matches
    spatial_matches: List[SpatialMatch] = field(default_factory=list)

    # Evidence chain
    chain_of_custody: Optional[ChainOfCustody] = None

    def sensor_count(self) -> int:
        """Number of distinct sensor types contributing to this detection."""
        return len(set(obs.sensor for obs in self.observations))

    def sensor_types(self) -> List[SensorType]:
        """List of distinct sensor types in this detection."""
        return sorted(set(obs.sensor for obs in self.observations), key=lambda s: s.value)

    def to_evidence_summary(self) -> dict:
        """Generate a summary suitable for evidence reports."""
        return {
            "detection_id": self.detection_id,
            "location": {"lon": self.location.lon, "lat": self.location.lat},
            "detection_time": self.detection_time.isoformat(),
            "corroboration_level": self.corroboration_level.value,
            "confidence": self.confidence.value,
            "confidence_score": round(self.confidence_score, 3),
            "sensor_count": self.sensor_count(),
            "sensors_used": [s.value for s in self.sensor_types()],
            "emission_rate_kg_h": self.emission_rate_kg_h,
            "emission_rate_uncertainty_kg_h": self.emission_rate_uncertainty_kg_h,
            "spectral_confirmed": self.spectral_confirmed,
            "area_flux_kg_km2_h": self.area_flux_kg_km2_h,
            "temporal_persistence_days": self.temporal_persistence_days,
            "chain_of_custody_case_id": (
                self.chain_of_custody.case_id if self.chain_of_custody else None
            ),
        }


def haversine_distance_m(coord_a: Coordinate, coord_b: Coordinate) -> float:
    """Compute great-circle distance between two WGS-84 points in meters."""
    R = 6_371_000  # Earth radius in meters
    lat1, lat2 = math.radians(coord_a.lat), math.radians(coord_b.lat)
    dlat = math.radians(coord_b.lat - coord_a.lat)
    dlon = math.radians(coord_b.lon - coord_a.lon)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bbox_center(bbox: BoundingBox) -> Coordinate:
    """Return the center point of a bounding box."""
    return Coordinate(
        lon=(bbox.min_lon + bbox.max_lon) / 2,
        lat=(bbox.min_lat + bbox.max_lat) / 2,
    )


def bboxes_overlap(a: BoundingBox, b: BoundingBox) -> bool:
    """Check if two bounding boxes overlap."""
    return (
        a.min_lon <= b.max_lon
        and a.max_lon >= b.min_lon
        and a.min_lat <= b.max_lat
        and a.max_lat >= b.min_lat
    )


class FusionEngine:
    """Multi-sensor methane detection fusion engine.

    Accepts observations from multiple sensors and produces corroborated
    FusedDetection objects with chain-of-custody evidence tracking.
    """

    # Spatial tolerance: maximum distance (meters) between sensor observations
    # to be considered co-located. Set conservatively for legal defensibility.
    SPATIAL_TOLERANCE_M: float = 500.0

    # Temporal tolerance: maximum time gap between sensor observations
    # to be considered contemporaneous.
    TEMPORAL_TOLERANCE: timedelta = timedelta(hours=72)

    # Minimum quality flag to accept an observation
    MIN_QUALITY: float = 0.5

    # Confidence scoring weights for each sensor type
    SENSOR_WEIGHTS: Dict[SensorType, float] = {
        SensorType.GHGSAT: 0.35,  # Highest spatial precision
        SensorType.TANAGER_1: 0.30,  # Spectral confirmation
        SensorType.METHANESAT: 0.20,  # Area-wide context
        SensorType.SENTINEL_5P: 0.15,  # Temporal persistence
    }

    # Minimum absorption depth to consider a Tanager-1 spectral confirmation
    TANAGER_CH4_THRESHOLD: float = 0.01

    def __init__(self) -> None:
        self._observations: Dict[SensorType, List[SensorObservation]] = {
            st: [] for st in SensorType
        }

    def ingest(
        self,
        observation: SensorObservation,
        chain: Optional[ChainOfCustody] = None,
        raw_bytes: Optional[bytes] = None,
    ) -> SensorObservation:
        """Ingest a sensor observation into the engine.

        Validates quality, optionally computes content hash, and records
        the ingest action in the chain of custody.
        """
        if observation.quality_flag < self.MIN_QUALITY:
            raise ValueError(
                f"Observation {observation.observation_id} quality "
                f"{observation.quality_flag} below minimum {self.MIN_QUALITY}"
            )

        # Compute content hash if raw data provided
        if raw_bytes is not None:
            observation.compute_content_hash(raw_bytes)

        self._observations[observation.sensor].append(observation)

        # Record in chain of custody
        if chain is not None:
            chain.add_link(
                action=CustodyAction.INGEST,
                actor=f"pmpe.fusion.engine.ingest",
                description=(
                    f"Ingested {observation.sensor.value} observation "
                    f"{observation.observation_id}"
                ),
                input_hashes=[observation.content_hash or "no-raw-data"],
                output_hash=hashlib.sha256(
                    json.dumps(observation.to_evidence_dict(), sort_keys=True).encode()
                ).hexdigest(),
                metadata={
                    "sensor": observation.sensor.value,
                    "observation_id": observation.observation_id,
                    "acquisition_time": observation.acquisition_time.isoformat(),
                    "quality_flag": observation.quality_flag,
                },
            )

        return observation

    def fuse(
        self,
        anchor: GHGSatObservation,
        chain: Optional[ChainOfCustody] = None,
    ) -> FusedDetection:
        """Fuse observations around a GHGSat anchor detection.

        GHGSat serves as the anchor because it has the highest spatial
        resolution (25m) and directly measures point-source emission rates.
        The engine then searches for corroborating observations from other
        sensors within spatial and temporal tolerances.

        Args:
            anchor: A GHGSat observation with a detected plume.
            chain: Optional chain of custody to record fusion steps.

        Returns:
            A FusedDetection with all corroborating evidence.
        """
        if not anchor.plume_detected or anchor.plume_center is None:
            raise ValueError(
                f"GHGSat observation {anchor.observation_id} has no plume detection"
            )

        detection = FusedDetection(
            detection_id=f"DET-{anchor.observation_id}",
            location=anchor.plume_center,
            detection_time=anchor.acquisition_time,
            observations=[anchor],
            emission_rate_kg_h=anchor.emission_rate_kg_h,
            emission_rate_uncertainty_kg_h=anchor.emission_rate_uncertainty_kg_h,
            emission_rate_source="ghgsat",
            chain_of_custody=chain,
        )

        # Search for corroborating Tanager-1 observations
        self._match_tanager(detection, anchor)

        # Search for corroborating MethaneSAT observations
        self._match_methanesat(detection, anchor)

        # Search for corroborating Sentinel-5P observations
        self._match_sentinel5p(detection, anchor)

        # Compute corroboration level and confidence
        self._compute_confidence(detection)

        # Record fusion in chain of custody
        if chain is not None:
            input_hashes = []
            for obs in detection.observations:
                h = obs.content_hash or hashlib.sha256(
                    obs.observation_id.encode()
                ).hexdigest()
                input_hashes.append(h)

            output_hash = hashlib.sha256(
                json.dumps(detection.to_evidence_summary(), sort_keys=True).encode()
            ).hexdigest()

            chain.add_link(
                action=CustodyAction.FUSE,
                actor="pmpe.fusion.engine.fuse",
                description=(
                    f"Fused {detection.sensor_count()} sensors into detection "
                    f"{detection.detection_id} — "
                    f"corroboration: {detection.corroboration_level.value}, "
                    f"confidence: {detection.confidence.value}"
                ),
                input_hashes=input_hashes,
                output_hash=output_hash,
                metadata=detection.to_evidence_summary(),
            )

        return detection

    def _match_tanager(
        self, detection: FusedDetection, anchor: GHGSatObservation
    ) -> None:
        """Find and match Tanager-1 observations to the GHGSat anchor."""
        for obs in self._observations[SensorType.TANAGER_1]:
            assert isinstance(obs, Tanager1Observation)

            # Check temporal proximity
            dt = abs((obs.acquisition_time - anchor.acquisition_time).total_seconds())
            if dt > self.TEMPORAL_TOLERANCE.total_seconds():
                continue

            # Check spatial overlap — plume within Tanager bbox OR center within tolerance
            obs_center = bbox_center(obs.bounding_box)
            distance = haversine_distance_m(anchor.plume_center, obs_center)
            point_in_bbox = self._point_in_bbox(anchor.plume_center, obs.bounding_box)

            match = SpatialMatch(
                obs_a_id=anchor.observation_id,
                obs_b_id=obs.observation_id,
                distance_m=distance,
                within_tolerance=point_in_bbox or distance <= self.SPATIAL_TOLERANCE_M,
            )
            detection.spatial_matches.append(match)

            if not match.within_tolerance:
                continue

            # Check spectral CH4 confirmation
            band_1650 = obs.methane_band_depth_1650nm or 0.0
            band_2300 = obs.methane_band_depth_2300nm or 0.0

            if band_1650 >= self.TANAGER_CH4_THRESHOLD or band_2300 >= self.TANAGER_CH4_THRESHOLD:
                detection.spectral_confirmed = True
                detection.ch4_band_depth_1650nm = obs.methane_band_depth_1650nm
                detection.ch4_band_depth_2300nm = obs.methane_band_depth_2300nm
                detection.observations.append(obs)

    def _match_methanesat(
        self, detection: FusedDetection, anchor: GHGSatObservation
    ) -> None:
        """Find and match MethaneSAT observations to the GHGSat anchor."""
        for obs in self._observations[SensorType.METHANESAT]:
            assert isinstance(obs, MethaneSATObservation)

            dt = abs((obs.acquisition_time - anchor.acquisition_time).total_seconds())
            if dt > self.TEMPORAL_TOLERANCE.total_seconds():
                continue

            # MethaneSAT has wide swath — check if anchor falls within bbox
            if not self._point_in_bbox(anchor.plume_center, obs.bounding_box):
                continue

            detection.area_flux_kg_km2_h = obs.area_flux_kg_km2_h
            detection.background_ch4_ppb = obs.background_ch4_ppb
            detection.enhancement_above_background_ppb = obs.enhancement_ppb
            detection.observations.append(obs)

    def _match_sentinel5p(
        self, detection: FusedDetection, anchor: GHGSatObservation
    ) -> None:
        """Find Sentinel-5P observations for temporal persistence context."""
        matching_s5p = []
        for obs in self._observations[SensorType.SENTINEL_5P]:
            assert isinstance(obs, Sentinel5PObservation)

            # Wider temporal window for persistence (30 days)
            dt = abs((obs.acquisition_time - anchor.acquisition_time).total_seconds())
            if dt > timedelta(days=30).total_seconds():
                continue

            # S5P has coarse resolution — check bbox overlap
            if not self._point_in_bbox(anchor.plume_center, obs.bounding_box):
                continue

            if obs.xch4_ppb is not None and obs.qa_value >= 0.5:
                matching_s5p.append(obs)

        if matching_s5p:
            # Sort by time to compute persistence
            matching_s5p.sort(key=lambda o: o.acquisition_time)
            earliest = matching_s5p[0].acquisition_time
            latest = matching_s5p[-1].acquisition_time
            persistence = (latest - earliest).days

            # Check for elevated CH4 — >1900 ppb indicates enhancement
            elevated = [o for o in matching_s5p if o.xch4_ppb > 1900]
            if elevated:
                detection.sentinel5p_corroborated = True
                detection.temporal_persistence_days = max(persistence, 1)
                # Add the best-quality S5P observation
                best = max(elevated, key=lambda o: o.qa_value)
                detection.observations.append(best)

    def _compute_confidence(self, detection: FusedDetection) -> None:
        """Compute corroboration level and confidence score."""
        n_sensors = detection.sensor_count()

        # Corroboration level
        level_map = {1: CorroborationLevel.SINGLE, 2: CorroborationLevel.DUAL,
                     3: CorroborationLevel.TRIPLE, 4: CorroborationLevel.QUAD}
        detection.corroboration_level = level_map.get(
            min(n_sensors, 4), CorroborationLevel.QUAD
        )

        # Confidence scoring
        score = 0.0
        sensor_types_present = set(obs.sensor for obs in detection.observations)

        for sensor_type, weight in self.SENSOR_WEIGHTS.items():
            if sensor_type in sensor_types_present:
                score += weight

        # Bonus for spectral confirmation
        if detection.spectral_confirmed:
            score += 0.05

        # Bonus for temporal persistence
        if detection.sentinel5p_corroborated and detection.temporal_persistence_days:
            if detection.temporal_persistence_days >= 7:
                score += 0.05
            if detection.temporal_persistence_days >= 14:
                score += 0.05

        # Penalty for high emission rate uncertainty
        if (
            detection.emission_rate_kg_h
            and detection.emission_rate_uncertainty_kg_h
            and detection.emission_rate_kg_h > 0
        ):
            uncertainty_ratio = (
                detection.emission_rate_uncertainty_kg_h / detection.emission_rate_kg_h
            )
            if uncertainty_ratio > 0.5:
                score -= 0.05

        detection.confidence_score = min(max(score, 0.0), 1.0)

        # Map to confidence level
        if detection.confidence_score >= 0.90:
            detection.confidence = FusionConfidence.VERY_HIGH
        elif detection.confidence_score >= 0.75:
            detection.confidence = FusionConfidence.HIGH
        elif detection.confidence_score >= 0.50:
            detection.confidence = FusionConfidence.MEDIUM
        else:
            detection.confidence = FusionConfidence.LOW

    @staticmethod
    def _point_in_bbox(point: Coordinate, bbox: BoundingBox) -> bool:
        """Check if a point falls within a bounding box."""
        return (
            bbox.min_lon <= point.lon <= bbox.max_lon
            and bbox.min_lat <= point.lat <= bbox.max_lat
        )
