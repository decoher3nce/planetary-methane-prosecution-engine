"""
Area-Wide Methane Emitter Scanner.

Scans a geographic region using all available sensor observations,
discovers all methane emission sources, fuses multi-sensor evidence
for each, and produces a prioritized prosecution queue ranked by
emission magnitude, evidentiary strength, and regulatory impact.

Usage:
    scanner = EmitterScanner()
    scanner.add_ghgsat_observations([...])
    scanner.add_tanager_observations([...])
    scanner.add_methanesat_observations([...])
    scanner.add_sentinel5p_observations([...])

    results = scanner.scan(region_bbox)
    queue = results.prioritize()

    # Prosecute the top emitters
    for target in queue[:10]:
        print(target.summary())
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

from src.evidence.chain_of_custody import ChainOfCustody
from src.fusion.engine import (
    CorroborationLevel,
    FusedDetection,
    FusionConfidence,
    FusionEngine,
    haversine_distance_m,
)
from src.sensors.models import (
    BoundingBox,
    Coordinate,
    GHGSatObservation,
    MethaneSATObservation,
    Sentinel5PObservation,
    SensorType,
    Tanager1Observation,
)


class ProsecutionPriority(str, Enum):
    """Prosecution priority tier."""

    CRITICAL = "critical"  # Prosecute immediately
    HIGH = "high"  # Prosecute within 30 days
    MEDIUM = "medium"  # Queue for enforcement
    LOW = "low"  # Monitor / issue notice
    WATCH = "watch"  # Insufficient evidence, keep monitoring


@dataclass
class EmitterTarget:
    """A discovered emitter ranked for prosecution.

    Combines the fused detection with a prosecution priority score
    that accounts for emission magnitude, evidentiary strength,
    and estimated regulatory impact.
    """

    detection: FusedDetection
    chain: ChainOfCustody
    priority: ProsecutionPriority = ProsecutionPriority.WATCH
    priority_score: float = 0.0

    # Scoring breakdown
    emission_score: float = 0.0  # 0-1 based on emission rate
    evidence_score: float = 0.0  # 0-1 based on corroboration
    persistence_score: float = 0.0  # 0-1 based on temporal persistence
    impact_score: float = 0.0  # 0-1 based on CO2e impact

    # Derived
    annualized_tonnes: float = 0.0
    co2e_tonnes_gwp20: float = 0.0

    def summary(self) -> dict:
        return {
            "detection_id": self.detection.detection_id,
            "location": {
                "lon": self.detection.location.lon,
                "lat": self.detection.location.lat,
            },
            "emission_rate_kg_h": self.detection.emission_rate_kg_h,
            "annualized_tonnes_ch4": round(self.annualized_tonnes, 1),
            "co2e_tonnes_gwp20": round(self.co2e_tonnes_gwp20, 0),
            "corroboration": self.detection.corroboration_level.value,
            "confidence": round(self.detection.confidence_score, 3),
            "sensors": self.detection.sensor_count(),
            "persistence_days": self.detection.temporal_persistence_days,
            "priority": self.priority.value,
            "priority_score": round(self.priority_score, 3),
            "score_breakdown": {
                "emission": round(self.emission_score, 3),
                "evidence": round(self.evidence_score, 3),
                "persistence": round(self.persistence_score, 3),
                "impact": round(self.impact_score, 3),
            },
        }


@dataclass
class ScanResults:
    """Results of an area-wide emitter scan."""

    region: BoundingBox
    scan_time: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    targets: List[EmitterTarget] = field(default_factory=list)
    total_ghgsat_plumes: int = 0
    total_fused: int = 0
    total_prosecutable: int = 0

    def prioritize(
        self,
        emission_weight: float = 0.40,
        evidence_weight: float = 0.25,
        persistence_weight: float = 0.15,
        impact_weight: float = 0.20,
    ) -> List[EmitterTarget]:
        """Score and rank all targets for prosecution priority.

        Weights control the relative importance of each factor:
          - emission_weight: Raw emission rate magnitude
          - evidence_weight: Multi-sensor corroboration strength
          - persistence_weight: How long the emission has persisted
          - impact_weight: Annualized CO2-equivalent impact

        Returns targets sorted by priority score (highest first).
        """
        if not self.targets:
            return []

        # Find max emission rate for normalization
        max_rate = max(
            (t.detection.emission_rate_kg_h or 0) for t in self.targets
        )
        max_rate = max(max_rate, 1.0)  # Avoid division by zero

        max_persistence = max(
            (t.detection.temporal_persistence_days or 0) for t in self.targets
        )
        max_persistence = max(max_persistence, 1)

        for target in self.targets:
            rate = target.detection.emission_rate_kg_h or 0

            # Emission score: normalized to max in this scan
            target.emission_score = rate / max_rate

            # Evidence score: based on corroboration level + confidence
            corr_scores = {
                CorroborationLevel.SINGLE: 0.2,
                CorroborationLevel.DUAL: 0.5,
                CorroborationLevel.TRIPLE: 0.8,
                CorroborationLevel.QUAD: 1.0,
            }
            target.evidence_score = (
                corr_scores.get(target.detection.corroboration_level, 0.1) * 0.6
                + target.detection.confidence_score * 0.4
            )

            # Persistence score
            days = target.detection.temporal_persistence_days or 0
            target.persistence_score = days / max_persistence

            # Impact score: annualized CO2e
            target.annualized_tonnes = rate * 8760 / 1000
            target.co2e_tonnes_gwp20 = target.annualized_tonnes * 80
            # Normalize: 100k tonnes CO2e/yr = score 1.0
            target.impact_score = min(target.co2e_tonnes_gwp20 / 100_000, 1.0)

            # Weighted composite
            target.priority_score = (
                target.emission_score * emission_weight
                + target.evidence_score * evidence_weight
                + target.persistence_score * persistence_weight
                + target.impact_score * impact_weight
            )

            # Map to priority tier
            if target.priority_score >= 0.75 and target.evidence_score >= 0.6:
                target.priority = ProsecutionPriority.CRITICAL
            elif target.priority_score >= 0.55 and target.evidence_score >= 0.4:
                target.priority = ProsecutionPriority.HIGH
            elif target.priority_score >= 0.35:
                target.priority = ProsecutionPriority.MEDIUM
            elif target.priority_score >= 0.15:
                target.priority = ProsecutionPriority.LOW
            else:
                target.priority = ProsecutionPriority.WATCH

        # Sort by priority score descending
        self.targets.sort(key=lambda t: t.priority_score, reverse=True)

        # Count prosecutable
        self.total_prosecutable = sum(
            1 for t in self.targets
            if t.priority in (ProsecutionPriority.CRITICAL, ProsecutionPriority.HIGH)
        )

        return self.targets

    def top(self, n: int = 10) -> List[EmitterTarget]:
        """Return the top N highest-priority targets."""
        return self.targets[:n]

    def by_priority(self, priority: ProsecutionPriority) -> List[EmitterTarget]:
        """Filter targets by priority tier."""
        return [t for t in self.targets if t.priority == priority]

    def stats(self) -> dict:
        """Summary statistics for the scan."""
        rates = [
            t.detection.emission_rate_kg_h
            for t in self.targets
            if t.detection.emission_rate_kg_h
        ]
        return {
            "region": {
                "min_lon": self.region.min_lon,
                "min_lat": self.region.min_lat,
                "max_lon": self.region.max_lon,
                "max_lat": self.region.max_lat,
            },
            "scan_time": self.scan_time.isoformat(),
            "total_plumes_detected": self.total_ghgsat_plumes,
            "total_fused_detections": self.total_fused,
            "total_prosecutable": self.total_prosecutable,
            "by_priority": {
                p.value: len(self.by_priority(p))
                for p in ProsecutionPriority
            },
            "emission_rates_kg_h": {
                "min": min(rates) if rates else 0,
                "max": max(rates) if rates else 0,
                "mean": sum(rates) / len(rates) if rates else 0,
                "total": sum(rates),
            },
            "total_annualized_co2e_tonnes": round(
                sum(t.co2e_tonnes_gwp20 for t in self.targets), 0
            ),
        }


class EmitterScanner:
    """Scans a region for methane emitters and builds a prosecution queue.

    Workflow:
      1. Load observations from all sensors
      2. Use GHGSat plumes as anchors (highest spatial resolution)
      3. Fuse each plume with corroborating sensors
      4. Score and rank for prosecution priority
    """

    # Cluster distance: GHGSat plumes within this distance are merged
    CLUSTER_DISTANCE_M: float = 500.0

    def __init__(self) -> None:
        self._ghgsat: List[GHGSatObservation] = []
        self._tanager: List[Tanager1Observation] = []
        self._methanesat: List[MethaneSATObservation] = []
        self._sentinel5p: List[Sentinel5PObservation] = []

    def add_ghgsat_observations(self, obs: List[GHGSatObservation]) -> None:
        self._ghgsat.extend(obs)

    def add_tanager_observations(self, obs: List[Tanager1Observation]) -> None:
        self._tanager.extend(obs)

    def add_methanesat_observations(self, obs: List[MethaneSATObservation]) -> None:
        self._methanesat.extend(obs)

    def add_sentinel5p_observations(self, obs: List[Sentinel5PObservation]) -> None:
        self._sentinel5p.extend(obs)

    def scan(
        self,
        region: BoundingBox,
        min_emission_rate_kg_h: float = 0.0,
    ) -> ScanResults:
        """Scan a region for all methane emitters.

        Args:
            region: Geographic bounding box to scan.
            min_emission_rate_kg_h: Minimum emission rate to include.

        Returns:
            ScanResults with all discovered emitters, ready for prioritization.
        """
        results = ScanResults(region=region)

        # Filter GHGSat plumes within region
        plumes = [
            obs for obs in self._ghgsat
            if obs.plume_detected
            and obs.plume_center is not None
            and self._in_region(obs.plume_center, region)
            and (obs.emission_rate_kg_h or 0) >= min_emission_rate_kg_h
        ]
        results.total_ghgsat_plumes = len(plumes)

        # Cluster nearby plumes (same source detected multiple times)
        clusters = self._cluster_plumes(plumes)

        # Fuse each cluster with corroborating sensors
        for anchor in clusters:
            engine = FusionEngine()
            chain = ChainOfCustody(
                description=f"Scan detection at {anchor.plume_center.lat:.4f}N, "
                f"{abs(anchor.plume_center.lon):.4f}W"
            )

            # Ingest the anchor
            engine.ingest(anchor, chain=chain)

            # Ingest all potentially corroborating observations
            for obs in self._tanager:
                try:
                    engine.ingest(obs, chain=chain)
                except ValueError:
                    pass  # Quality too low

            for obs in self._methanesat:
                try:
                    engine.ingest(obs, chain=chain)
                except ValueError:
                    pass

            for obs in self._sentinel5p:
                try:
                    engine.ingest(obs, chain=chain)
                except ValueError:
                    pass

            # Fuse
            detection = engine.fuse(anchor, chain=chain)
            results.targets.append(
                EmitterTarget(detection=detection, chain=chain)
            )

        results.total_fused = len(results.targets)
        return results

    def _cluster_plumes(
        self, plumes: List[GHGSatObservation]
    ) -> List[GHGSatObservation]:
        """Cluster nearby plumes, keeping the highest-rate observation per cluster."""
        if not plumes:
            return []

        # Sort by emission rate descending so we keep the strongest
        sorted_plumes = sorted(
            plumes,
            key=lambda p: p.emission_rate_kg_h or 0,
            reverse=True,
        )

        representatives: List[GHGSatObservation] = []
        used = set()

        for plume in sorted_plumes:
            if id(plume) in used:
                continue

            # This plume becomes a cluster representative
            representatives.append(plume)
            used.add(id(plume))

            # Mark all nearby plumes as used
            for other in sorted_plumes:
                if id(other) in used:
                    continue
                dist = haversine_distance_m(plume.plume_center, other.plume_center)
                if dist <= self.CLUSTER_DISTANCE_M:
                    used.add(id(other))

        return representatives

    @staticmethod
    def _in_region(point: Coordinate, region: BoundingBox) -> bool:
        return (
            region.min_lon <= point.lon <= region.max_lon
            and region.min_lat <= point.lat <= region.max_lat
        )
