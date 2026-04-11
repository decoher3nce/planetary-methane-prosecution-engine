"""Tests for area-wide emitter scanning and prioritization."""

from datetime import datetime, timedelta, timezone

import pytest

from src.scanning.scanner import (
    EmitterScanner,
    EmitterTarget,
    ProsecutionPriority,
    ScanResults,
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

NOW = datetime(2026, 3, 15, 14, 30, 0, tzinfo=timezone.utc)
PERMIAN = BoundingBox(min_lon=-104.5, min_lat=31.0, max_lon=-103.0, max_lat=32.5)


def _ghgsat(
    lon: float,
    lat: float,
    rate: float,
    obs_id: str,
    time_offset: timedelta = timedelta(0),
) -> GHGSatObservation:
    return GHGSatObservation(
        observation_id=obs_id,
        acquisition_time=NOW + time_offset,
        processing_level=ProcessingLevel.L2,
        bounding_box=BoundingBox(
            min_lon=lon - 0.05, min_lat=lat - 0.05,
            max_lon=lon + 0.05, max_lat=lat + 0.05,
        ),
        data_uri=f"s3://ghgsat/{obs_id}.nc",
        quality_flag=0.9,
        plume_detected=True,
        plume_center=Coordinate(lon=lon, lat=lat),
        emission_rate_kg_h=rate,
        emission_rate_uncertainty_kg_h=rate * 0.15,
    )


def _tanager(lon: float, lat: float, obs_id: str) -> Tanager1Observation:
    return Tanager1Observation(
        observation_id=obs_id,
        acquisition_time=NOW + timedelta(hours=2),
        processing_level=ProcessingLevel.L2,
        bounding_box=BoundingBox(
            min_lon=lon - 0.05, min_lat=lat - 0.05,
            max_lon=lon + 0.05, max_lat=lat + 0.05,
        ),
        data_uri=f"s3://tanager/{obs_id}.nc",
        quality_flag=0.9,
        methane_band_depth_1650nm=0.04,
        methane_band_depth_2300nm=0.03,
        swir_snr=200.0,
    )


def _methanesat() -> MethaneSATObservation:
    return MethaneSATObservation(
        observation_id="MSAT-SCAN-001",
        acquisition_time=NOW + timedelta(hours=4),
        processing_level=ProcessingLevel.L2,
        bounding_box=PERMIAN,
        data_uri="s3://methanesat/scan.nc",
        quality_flag=0.85,
        area_flux_kg_km2_h=8.0,
        enhancement_ppb=30.0,
        background_ch4_ppb=1900.0,
    )


def _sentinel5p(time_offset: timedelta = timedelta(days=-3)) -> Sentinel5PObservation:
    return Sentinel5PObservation(
        observation_id=f"S5P-SCAN-{abs(time_offset.days):03d}",
        acquisition_time=NOW + time_offset,
        processing_level=ProcessingLevel.L2,
        bounding_box=PERMIAN,
        data_uri=f"s3://s5p/scan_{abs(time_offset.days)}.nc",
        quality_flag=0.7,
        xch4_ppb=1940.0,
        qa_value=0.75,
        cloud_fraction=0.05,
    )


class TestEmitterScanner:
    def test_scan_finds_plumes(self):
        scanner = EmitterScanner()
        scanner.add_ghgsat_observations([
            _ghgsat(-103.7, 31.8, 500, "GHG-001"),
            _ghgsat(-103.5, 31.5, 200, "GHG-002"),
            _ghgsat(-104.0, 32.0, 1000, "GHG-003"),
        ])

        results = scanner.scan(PERMIAN)
        assert results.total_ghgsat_plumes == 3
        assert results.total_fused == 3
        assert len(results.targets) == 3

    def test_scan_filters_by_region(self):
        scanner = EmitterScanner()
        scanner.add_ghgsat_observations([
            _ghgsat(-103.7, 31.8, 500, "GHG-IN"),
            _ghgsat(-100.0, 35.0, 2000, "GHG-OUT"),  # Outside region
        ])

        results = scanner.scan(PERMIAN)
        assert results.total_ghgsat_plumes == 1

    def test_scan_filters_by_min_rate(self):
        scanner = EmitterScanner()
        scanner.add_ghgsat_observations([
            _ghgsat(-103.7, 31.8, 500, "GHG-BIG"),
            _ghgsat(-103.5, 31.5, 50, "GHG-SMALL"),
        ])

        results = scanner.scan(PERMIAN, min_emission_rate_kg_h=100)
        assert results.total_ghgsat_plumes == 1

    def test_clusters_nearby_plumes(self):
        scanner = EmitterScanner()
        # Two plumes very close — should cluster into one
        scanner.add_ghgsat_observations([
            _ghgsat(-103.70, 31.80, 500, "GHG-A"),
            _ghgsat(-103.701, 31.801, 300, "GHG-B"),  # ~130m away
        ])

        results = scanner.scan(PERMIAN)
        # Should cluster to 1, keeping the 500 kg/hr one
        assert len(results.targets) == 1
        assert results.targets[0].detection.emission_rate_kg_h == 500

    def test_keeps_distant_plumes_separate(self):
        scanner = EmitterScanner()
        scanner.add_ghgsat_observations([
            _ghgsat(-103.70, 31.80, 500, "GHG-A"),
            _ghgsat(-103.50, 31.50, 300, "GHG-B"),  # ~40km away
        ])

        results = scanner.scan(PERMIAN)
        assert len(results.targets) == 2


class TestMultiSensorScan:
    def test_scan_with_corroboration(self):
        scanner = EmitterScanner()
        scanner.add_ghgsat_observations([
            _ghgsat(-103.7, 31.8, 500, "GHG-001"),
        ])
        scanner.add_tanager_observations([
            _tanager(-103.7, 31.8, "TAN-001"),
        ])
        scanner.add_methanesat_observations([_methanesat()])
        scanner.add_sentinel5p_observations([
            _sentinel5p(timedelta(days=-1)),
            _sentinel5p(timedelta(days=-7)),
            _sentinel5p(timedelta(days=-14)),
        ])

        results = scanner.scan(PERMIAN)
        results.prioritize()

        assert len(results.targets) == 1
        target = results.targets[0]
        assert target.detection.sensor_count() >= 3
        assert target.detection.spectral_confirmed is True


class TestPrioritization:
    def test_ranked_by_emission_rate(self):
        scanner = EmitterScanner()
        scanner.add_ghgsat_observations([
            _ghgsat(-103.7, 31.8, 100, "GHG-SMALL"),
            _ghgsat(-103.5, 31.5, 2000, "GHG-HUGE"),
            _ghgsat(-104.0, 32.0, 500, "GHG-MED"),
        ])

        results = scanner.scan(PERMIAN)
        queue = results.prioritize()

        # Highest emitter should be first
        assert queue[0].detection.emission_rate_kg_h == 2000
        assert queue[1].detection.emission_rate_kg_h == 500
        assert queue[2].detection.emission_rate_kg_h == 100

    def test_priority_tiers(self):
        scanner = EmitterScanner()
        scanner.add_ghgsat_observations([
            _ghgsat(-103.7, 31.8, 3000, "GHG-HUGE"),
            _ghgsat(-103.5, 31.5, 50, "GHG-TINY"),
        ])
        # Add corroboration for the big one
        scanner.add_tanager_observations([
            _tanager(-103.7, 31.8, "TAN-001"),
        ])
        scanner.add_methanesat_observations([_methanesat()])

        results = scanner.scan(PERMIAN)
        queue = results.prioritize()

        big = queue[0]
        small = queue[-1]

        assert big.priority in (ProsecutionPriority.CRITICAL, ProsecutionPriority.HIGH)
        assert big.priority_score > small.priority_score

    def test_co2e_calculation(self):
        scanner = EmitterScanner()
        scanner.add_ghgsat_observations([
            _ghgsat(-103.7, 31.8, 500, "GHG-001"),
        ])

        results = scanner.scan(PERMIAN)
        results.prioritize()

        target = results.targets[0]
        # 500 kg/hr * 8760 hr/yr / 1000 = 4380 tonnes/yr
        assert target.annualized_tonnes == pytest.approx(4380.0)
        # 4380 * 80 = 350,400 tonnes CO2e
        assert target.co2e_tonnes_gwp20 == pytest.approx(350_400.0)

    def test_evidence_boosts_priority(self):
        """Same emission rate, but better evidence should rank higher."""
        scanner = EmitterScanner()
        scanner.add_ghgsat_observations([
            _ghgsat(-103.7, 31.8, 500, "GHG-STRONG"),
            _ghgsat(-103.5, 31.5, 500, "GHG-WEAK"),
        ])
        # Only corroborate the first one
        scanner.add_tanager_observations([
            _tanager(-103.7, 31.8, "TAN-001"),
        ])
        scanner.add_methanesat_observations([_methanesat()])

        results = scanner.scan(PERMIAN)
        queue = results.prioritize()

        strong = next(t for t in queue if t.detection.detection_id == "DET-GHG-STRONG")
        weak = next(t for t in queue if t.detection.detection_id == "DET-GHG-WEAK")

        assert strong.evidence_score > weak.evidence_score
        assert strong.priority_score > weak.priority_score


class TestScanResults:
    def test_top_n(self):
        scanner = EmitterScanner()
        scanner.add_ghgsat_observations([
            _ghgsat(-103.7, 31.8, 500, "GHG-001"),
            _ghgsat(-103.5, 31.5, 200, "GHG-002"),
            _ghgsat(-104.0, 32.0, 1000, "GHG-003"),
        ])

        results = scanner.scan(PERMIAN)
        results.prioritize()

        top2 = results.top(2)
        assert len(top2) == 2
        assert top2[0].priority_score >= top2[1].priority_score

    def test_filter_by_priority(self):
        scanner = EmitterScanner()
        scanner.add_ghgsat_observations([
            _ghgsat(-103.7, 31.8, 3000, "GHG-BIG"),
            _ghgsat(-103.5, 31.5, 50, "GHG-SMALL"),
        ])
        scanner.add_tanager_observations([_tanager(-103.7, 31.8, "TAN-001")])
        scanner.add_methanesat_observations([_methanesat()])

        results = scanner.scan(PERMIAN)
        results.prioritize()

        critical = results.by_priority(ProsecutionPriority.CRITICAL)
        # The big one with corroboration should be critical
        assert any(
            t.detection.emission_rate_kg_h == 3000 for t in critical
        ) or len(results.by_priority(ProsecutionPriority.HIGH)) > 0

    def test_stats(self):
        scanner = EmitterScanner()
        scanner.add_ghgsat_observations([
            _ghgsat(-103.7, 31.8, 500, "GHG-001"),
            _ghgsat(-103.5, 31.5, 1000, "GHG-002"),
        ])

        results = scanner.scan(PERMIAN)
        results.prioritize()

        stats = results.stats()
        assert stats["total_fused_detections"] == 2
        assert stats["emission_rates_kg_h"]["max"] == 1000
        assert stats["emission_rates_kg_h"]["total"] == 1500
        assert stats["total_annualized_co2e_tonnes"] > 0

    def test_target_summary(self):
        scanner = EmitterScanner()
        scanner.add_ghgsat_observations([
            _ghgsat(-103.7, 31.8, 500, "GHG-001"),
        ])

        results = scanner.scan(PERMIAN)
        results.prioritize()

        s = results.targets[0].summary()
        assert s["emission_rate_kg_h"] == 500
        assert "score_breakdown" in s
        assert s["priority"] in [p.value for p in ProsecutionPriority]
