"""
Example: Prosecute a Permian Basin super-emitter.

Run with: python3 example_prosecution.py

This walks through the full pipeline — from raw sensor observations
to a sealed court-ready evidence package on disk.
"""

from datetime import datetime, timedelta, timezone

from src.evidence.chain_of_custody import ChainOfCustody
from src.evidence.packager import EvidencePackager
from src.facilities.identifier import (
    CandidateFacility,
    FacilityIdentifier,
    SpectralIndices,
)
from src.facilities.models import InfrastructureSignature
from src.fusion.engine import FusionEngine
from src.reports.attribution import ReportGenerator
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


def main():
    # ================================================================
    # 1. START A CASE — initialize chain of custody
    # ================================================================
    chain = ChainOfCustody(
        description="Permian Basin super-emitter — suspected well pad near Pecos, TX"
    )
    print(f"Case opened: {chain.case_id}")
    print(f"  Description: {chain.description}\n")

    # ================================================================
    # 2. INGEST SENSOR OBSERVATIONS
    # ================================================================
    engine = FusionEngine()

    # GHGSat: detected a plume at 25m resolution
    ghgsat = GHGSatObservation(
        observation_id="GHG-2026-0315-PB001",
        acquisition_time=NOW,
        processing_level=ProcessingLevel.L2,
        bounding_box=BoundingBox(min_lon=-103.8, min_lat=31.7, max_lon=-103.6, max_lat=31.9),
        data_uri="s3://ghgsat/permian/2026-03-15/obs_001.nc",
        quality_flag=0.95,
        plume_detected=True,
        plume_center=Coordinate(lon=-103.72, lat=31.81),
        emission_rate_kg_h=500.0,
        emission_rate_uncertainty_kg_h=75.0,
    )

    # Tanager-1: 400-band hyperspectral confirms CH4 absorption
    tanager = Tanager1Observation(
        observation_id="TAN-2026-0315-PB001",
        acquisition_time=NOW + timedelta(hours=2),
        processing_level=ProcessingLevel.L2,
        bounding_box=BoundingBox(min_lon=-103.8, min_lat=31.7, max_lon=-103.6, max_lat=31.9),
        data_uri="s3://tanager/permian/2026-03-15/scene_001.nc",
        quality_flag=0.92,
        methane_band_depth_1650nm=0.045,
        methane_band_depth_2300nm=0.032,
        swir_snr=250.0,
    )

    # MethaneSAT: area-wide quantification at basin scale
    methanesat = MethaneSATObservation(
        observation_id="MSAT-2026-0315-PB001",
        acquisition_time=NOW + timedelta(hours=6),
        processing_level=ProcessingLevel.L2,
        bounding_box=BoundingBox(min_lon=-104.5, min_lat=31.0, max_lon=-103.0, max_lat=32.5),
        data_uri="s3://methanesat/permian/2026-03-15/swath_001.nc",
        quality_flag=0.88,
        area_flux_kg_km2_h=12.5,
        enhancement_ppb=45.0,
        background_ch4_ppb=1900.0,
    )

    # Sentinel-5P: temporal persistence over 2 weeks
    s5p_observations = [
        Sentinel5PObservation(
            observation_id=f"S5P-2026-{(NOW - timedelta(days=d)).strftime('%m%d')}-PB001",
            acquisition_time=NOW - timedelta(days=d),
            processing_level=ProcessingLevel.L2,
            bounding_box=BoundingBox(min_lon=-104.5, min_lat=31.0, max_lon=-103.0, max_lat=32.5),
            data_uri=f"s3://sentinel5p/CH4/2026-{(NOW - timedelta(days=d)).strftime('%m-%d')}.nc",
            quality_flag=0.7,
            xch4_ppb=1920 + d * 2,  # Elevated readings over time
            qa_value=0.75,
            cloud_fraction=0.05,
        )
        for d in [1, 5, 10, 14]
    ]

    print("Ingesting sensor observations...")
    for obs in [ghgsat, tanager, methanesat] + s5p_observations:
        engine.ingest(obs, chain=chain, raw_bytes=f"raw-{obs.observation_id}".encode())
        print(f"  [{obs.sensor.value}] {obs.observation_id}")

    print(f"\n  Chain of custody: {len(chain.links)} links\n")

    # ================================================================
    # 3. FUSE INTO CORROBORATED DETECTION
    # ================================================================
    print("Fusing multi-sensor detection...")
    detection = engine.fuse(ghgsat, chain=chain)

    print(f"  Detection ID:      {detection.detection_id}")
    print(f"  Sensors fused:     {detection.sensor_count()} ({', '.join(s.value for s in detection.sensor_types())})")
    print(f"  Corroboration:     {detection.corroboration_level.value.upper()}")
    print(f"  Confidence:        {detection.confidence.value} ({detection.confidence_score:.1%})")
    print(f"  Emission rate:     {detection.emission_rate_kg_h} ± {detection.emission_rate_uncertainty_kg_h} kg CH4/hr")
    print(f"  Spectral confirm:  {detection.spectral_confirmed}")
    print(f"  Area flux:         {detection.area_flux_kg_km2_h} kg/km²/hr")
    print(f"  S5P persistence:   {detection.temporal_persistence_days} days")
    print(f"\n  Chain of custody: {len(chain.links)} links\n")

    # ================================================================
    # 4. IDENTIFY RESPONSIBLE FACILITY (Sentinel-2)
    # ================================================================
    print("Identifying facilities from Sentinel-2 imagery...")
    identifier = FacilityIdentifier()

    # In production, these come from Sentinel-2 raster analysis.
    # Here we provide pre-extracted candidates.
    candidates = [
        CandidateFacility(
            centroid=Coordinate(lon=-103.723, lat=31.808),
            bbox=BoundingBox(min_lon=-103.724, min_lat=31.807, max_lon=-103.722, max_lat=31.809),
            area_m2=5000,
            perimeter_m=300,
            orientation_deg=45.0,
            spectral=SpectralIndices(ndvi=0.08, ndbi=0.12, bsi=0.3, ndwi=-0.2),
            signatures=[
                InfrastructureSignature.CLEARED_PAD,
                InfrastructureSignature.ROAD_ACCESS,
                InfrastructureSignature.CIRCULAR_TANK,
            ],
        ),
        CandidateFacility(
            centroid=Coordinate(lon=-103.71, lat=31.815),
            bbox=BoundingBox(min_lon=-103.712, min_lat=31.813, max_lon=-103.708, max_lat=31.817),
            area_m2=12000,
            perimeter_m=450,
            orientation_deg=90.0,
            spectral=SpectralIndices(ndvi=0.05, ndbi=0.2, bsi=0.25, ndwi=-0.15),
            signatures=[
                InfrastructureSignature.THERMAL_ANOMALY,
                InfrastructureSignature.CLEARED_PAD,
            ],
        ),
    ]

    facilities = identifier.identify_facilities(
        detection,
        candidates=candidates,
        wind_direction_deg=225.0,  # Wind from SW
        chain=chain,
    )

    for i, f in enumerate(facilities):
        tag = "PRIMARY" if i == 0 else f"ALT #{i}"
        print(f"  [{tag}] {f.facility_type.value} — confidence: {f.identification_confidence:.1%}")
        print(f"           distance: {f.distance_to_plume_m:.0f}m, upwind: {f.upwind_of_plume}")

    primary = facilities[0]
    print(f"\n  Chain of custody: {len(chain.links)} links\n")

    # ================================================================
    # 5. GENERATE ATTRIBUTION REPORT
    # ================================================================
    print("Generating attribution report...")
    gen = ReportGenerator()
    report = gen.generate(detection, primary, chain)

    print(f"  Report ID:         {report.report_id}")
    print(f"  Severity:          {report.regulatory['severity'].upper()}")
    print(f"  Super-emitter:     {report.regulatory['qualifies_as_super_emitter']}")
    print(f"  Violations:        {', '.join(report.regulatory['applicable_violations'])}")
    print(f"  Est. penalty:      ${report.regulatory['estimated_penalty_usd']:,.0f}")
    print(f"  Annualized:        {report.emission['rate_tonnes_year']:,.1f} tonnes CH4/yr")
    print(f"  CO2-equivalent:    {report.emission['rate_co2e_tonnes_year']:,.0f} tonnes CO2e/yr ({report.emission['gwp_basis']})")
    print(f"\n  Chain of custody: {len(chain.links)} links\n")

    # ================================================================
    # 6. PACKAGE AS COURT-READY EVIDENCE
    # ================================================================
    print("Packaging court-ready evidence...")
    packager = EvidencePackager()
    package = packager.package(
        report, chain,
        additional_artifacts={
            "plume_visualization.png": open("fusion_visualization.png", "rb").read()
        } if __import__("os").path.exists("fusion_visualization.png") else None,
    )

    # Write to disk
    out_dir = package.write_to_directory("evidence_packages")

    print(f"  Package ID:        {package.package_id}")
    print(f"  Artifacts:         {len(package.artifacts)}")
    print(f"  Seal:              {package.seal[:32]}...")
    print(f"  Seal valid:        {package.verify_seal()}")
    print(f"  Artifact errors:   {package.verify_artifacts()}")
    print(f"  Written to:        {out_dir}/")
    print()

    # ================================================================
    # 7. VERIFY CHAIN OF CUSTODY
    # ================================================================
    valid, errors = chain.verify_integrity()
    print(f"Chain of custody: {len(chain.links)} links — {'VERIFIED' if valid else 'BROKEN'}")
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
    else:
        print("  All hashes valid. Tamper-evident chain intact from ingest to packaging.")

    print()
    for step in chain.get_audit_trail():
        print(f"  {step['step']:2d}. [{step['action']:>9s}] {step['description'][:80]}")

    print(f"\n{'='*72}")
    print(f"  EVIDENCE PACKAGE READY FOR LEGAL SUBMISSION")
    print(f"  Case: {chain.case_id}")
    print(f"  Output: {out_dir}/")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
