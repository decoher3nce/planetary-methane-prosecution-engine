"""
Example: Scan the Permian Basin for all super-emitters and prioritize for prosecution.

Run with: python3 example_scan.py

Simulates loading observations from all sensors across the basin,
discovering every methane plume, fusing multi-sensor evidence,
and producing a ranked prosecution queue.
"""

import random
from datetime import datetime, timedelta, timezone

from src.scanning.scanner import EmitterScanner, ProsecutionPriority
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

random.seed(42)


def generate_synthetic_basin():
    """Generate a realistic set of emitters across the Permian Basin."""

    ghgsat_obs = []
    tanager_obs = []

    # Simulate 25 detected plumes across the basin
    emitters = [
        # (lon, lat, rate_kg_h, name)
        (-103.72, 31.81, 2500, "Mega-emitter wellpad cluster"),
        (-103.45, 31.55, 1800, "Abandoned well complex"),
        (-104.10, 32.20, 1200, "Compressor station leak"),
        (-103.88, 31.92, 950, "Processing plant venting"),
        (-103.30, 31.25, 800, "Pipeline rupture"),
        (-103.95, 32.35, 650, "Tank battery emissions"),
        (-104.20, 31.80, 500, "Active drilling pad"),
        (-103.60, 31.40, 420, "Flare malfunction"),
        (-104.35, 32.10, 350, "Gathering line leak"),
        (-103.15, 31.15, 280, "Wellhead casing failure"),
        (-103.78, 32.45, 220, "Storage facility"),
        (-104.00, 31.50, 180, "Separator leak"),
        (-103.55, 32.00, 150, "Pneumatic controller"),
        (-104.40, 31.30, 120, "Dehydrator emissions"),
        (-103.25, 31.70, 100, "Pump jack leak"),
        (-103.90, 31.20, 85, "Small wellpad"),
        (-104.15, 32.40, 70, "Minor leak #1"),
        (-103.40, 31.85, 60, "Minor leak #2"),
        (-103.70, 31.35, 45, "Small vent"),
        (-104.30, 32.25, 35, "Trace emission #1"),
        (-103.50, 31.60, 25, "Trace emission #2"),
        (-103.80, 32.15, 20, "Background source"),
        (-104.05, 31.70, 15, "Marginal detection"),
        (-103.35, 31.45, 10, "Near detection limit"),
        (-103.65, 32.30, 8, "Borderline detection"),
    ]

    for i, (lon, lat, rate, name) in enumerate(emitters):
        ghgsat_obs.append(GHGSatObservation(
            observation_id=f"GHG-SCAN-{i+1:03d}",
            acquisition_time=NOW + timedelta(hours=random.uniform(-12, 12)),
            processing_level=ProcessingLevel.L2,
            bounding_box=BoundingBox(
                min_lon=lon - 0.05, min_lat=lat - 0.05,
                max_lon=lon + 0.05, max_lat=lat + 0.05,
            ),
            data_uri=f"s3://ghgsat/permian/scan/{i+1:03d}.nc",
            quality_flag=random.uniform(0.7, 0.98),
            plume_detected=True,
            plume_center=Coordinate(lon=lon, lat=lat),
            emission_rate_kg_h=rate,
            emission_rate_uncertainty_kg_h=rate * random.uniform(0.1, 0.2),
        ))

        # Tanager-1 coverage for ~60% of emitters (not all have overpasses)
        if random.random() < 0.6:
            tanager_obs.append(Tanager1Observation(
                observation_id=f"TAN-SCAN-{i+1:03d}",
                acquisition_time=NOW + timedelta(hours=random.uniform(0, 6)),
                processing_level=ProcessingLevel.L2,
                bounding_box=BoundingBox(
                    min_lon=lon - 0.05, min_lat=lat - 0.05,
                    max_lon=lon + 0.05, max_lat=lat + 0.05,
                ),
                data_uri=f"s3://tanager/permian/scan/{i+1:03d}.nc",
                quality_flag=random.uniform(0.8, 0.95),
                methane_band_depth_1650nm=0.01 + rate / 50000,
                methane_band_depth_2300nm=0.008 + rate / 60000,
                swir_snr=random.uniform(150, 300),
            ))

    # MethaneSAT: basin-wide swath
    methanesat_obs = [MethaneSATObservation(
        observation_id="MSAT-SCAN-001",
        acquisition_time=NOW + timedelta(hours=4),
        processing_level=ProcessingLevel.L2,
        bounding_box=PERMIAN,
        data_uri="s3://methanesat/permian/scan/001.nc",
        quality_flag=0.88,
        area_flux_kg_km2_h=8.5,
        enhancement_ppb=35.0,
        background_ch4_ppb=1900.0,
    )]

    # Sentinel-5P: daily passes over 3 weeks
    s5p_obs = [
        Sentinel5PObservation(
            observation_id=f"S5P-SCAN-{d:03d}",
            acquisition_time=NOW - timedelta(days=d),
            processing_level=ProcessingLevel.L2,
            bounding_box=PERMIAN,
            data_uri=f"s3://s5p/permian/scan/{d:03d}.nc",
            quality_flag=0.7,
            xch4_ppb=1910 + random.uniform(0, 50),
            qa_value=random.uniform(0.6, 0.85),
            cloud_fraction=random.uniform(0.0, 0.2),
        )
        for d in range(1, 22)
    ]

    return ghgsat_obs, tanager_obs, methanesat_obs, s5p_obs


def main():
    print("=" * 72)
    print("  PLANETARY METHANE PROSECUTION ENGINE")
    print("  Area-Wide Emitter Scan — Permian Basin, TX/NM")
    print("=" * 72)
    print()

    # ── Load sensor data ─────────────────────────────────────────
    ghgsat, tanager, methanesat, s5p = generate_synthetic_basin()

    scanner = EmitterScanner()
    scanner.add_ghgsat_observations(ghgsat)
    scanner.add_tanager_observations(tanager)
    scanner.add_methanesat_observations(methanesat)
    scanner.add_sentinel5p_observations(s5p)

    print(f"Loaded observations:")
    print(f"  GHGSat plumes:    {len(ghgsat)}")
    print(f"  Tanager-1 scenes: {len(tanager)}")
    print(f"  MethaneSAT swaths:{len(methanesat)}")
    print(f"  Sentinel-5P days: {len(s5p)}")
    print()

    # ── Scan the basin ───────────────────────────────────────────
    print("Scanning Permian Basin for methane emitters...")
    results = scanner.scan(PERMIAN)
    print(f"  Plumes detected:  {results.total_ghgsat_plumes}")
    print(f"  Fused detections: {results.total_fused}")
    print()

    # ── Prioritize for prosecution ───────────────────────────────
    print("Prioritizing for prosecution...")
    queue = results.prioritize()
    print()

    # ── Print the prosecution queue ──────────────────────────────
    stats = results.stats()
    print(f"{'='*72}")
    print(f"  PROSECUTION QUEUE — {stats['total_fused_detections']} emitters ranked")
    print(f"  Total basin emissions: {stats['emission_rates_kg_h']['total']:,.0f} kg CH4/hr")
    print(f"  Annualized CO2e:       {stats['total_annualized_co2e_tonnes']:,.0f} tonnes (GWP-20)")
    print(f"{'='*72}")
    print()

    # By priority tier
    for tier in ProsecutionPriority:
        tier_targets = results.by_priority(tier)
        if not tier_targets:
            continue
        tier_emissions = sum(t.detection.emission_rate_kg_h or 0 for t in tier_targets)
        print(f"  {tier.value.upper()} ({len(tier_targets)} emitters, {tier_emissions:,.0f} kg/hr total):")
        for t in tier_targets:
            s = t.summary()
            print(
                f"    {s['detection_id']:<28s} "
                f"{s['emission_rate_kg_h']:>6,.0f} kg/hr  "
                f"{s['co2e_tonnes_gwp20']:>10,.0f} tCO2e/yr  "
                f"corr:{s['corroboration']:>6s}  "
                f"conf:{s['confidence']:>5.1%}  "
                f"score:{s['priority_score']:.3f}"
            )
        print()

    # ── Top 5 summary ────────────────────────────────────────────
    print(f"{'='*72}")
    print("  TOP 5 PROSECUTION TARGETS")
    print(f"{'='*72}")
    for i, target in enumerate(results.top(5)):
        s = target.summary()
        print(f"""
  #{i+1}  {s['detection_id']}
      Location:       {s['location']['lat']:.4f}°N, {abs(s['location']['lon']):.4f}°W
      Emission rate:  {s['emission_rate_kg_h']:,.0f} kg CH4/hr
      Annualized:     {s['annualized_tonnes_ch4']:,.1f} tonnes CH4/yr
      CO2-equivalent: {s['co2e_tonnes_gwp20']:,.0f} tonnes CO2e/yr (GWP-20)
      Corroboration:  {s['corroboration'].upper()} ({s['sensors']} sensors)
      Confidence:     {s['confidence']:.1%}
      Persistence:    {s['persistence_days'] or 'N/A'} days
      Priority:       {s['priority'].upper()} (score: {s['priority_score']:.3f})
      Scores:         emission={s['score_breakdown']['emission']:.3f}  evidence={s['score_breakdown']['evidence']:.3f}  persistence={s['score_breakdown']['persistence']:.3f}  impact={s['score_breakdown']['impact']:.3f}""")

    print(f"\n{'='*72}")
    print(f"  {stats['total_prosecutable']} emitters ready for immediate prosecution")
    print(f"  Use example_prosecution.py pipeline for each target")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
