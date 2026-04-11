"""
Methane Prosecution Viewer — Flask backend.

Serves an interactive map UI for scanning regions, viewing emitter
prioritization, and visualizing individual plumes with multi-sensor context.

Run with: python3 viewer.py
"""

import io
import json
import random
from datetime import datetime, timedelta, timezone

from flask import Flask, Response, jsonify, render_template, request

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

app = Flask(__name__)

# ── Synthetic data generator ─────────────────────────────────────────

NOW = datetime(2026, 3, 15, 14, 30, 0, tzinfo=timezone.utc)


def generate_basin_data(bbox, n_emitters=30, seed=42):
    """Generate realistic synthetic emitter data within a bounding box."""
    rng = random.Random(seed)

    min_lon, min_lat = bbox[0], bbox[1]
    max_lon, max_lat = bbox[2], bbox[3]

    # Power-law-ish distribution: few huge emitters, many small ones
    rates = sorted(
        [rng.paretovariate(1.2) * 80 for _ in range(n_emitters)],
        reverse=True,
    )
    # Cap at realistic max
    rates = [min(r, 5000) for r in rates]

    ghgsat_obs = []
    tanager_obs = []

    for i, rate in enumerate(rates):
        lon = rng.uniform(min_lon + 0.05, max_lon - 0.05)
        lat = rng.uniform(min_lat + 0.05, max_lat - 0.05)

        ghgsat_obs.append(GHGSatObservation(
            observation_id=f"GHG-SCAN-{i+1:03d}",
            acquisition_time=NOW + timedelta(hours=rng.uniform(-12, 12)),
            processing_level=ProcessingLevel.L2,
            bounding_box=BoundingBox(
                min_lon=lon - 0.05, min_lat=lat - 0.05,
                max_lon=lon + 0.05, max_lat=lat + 0.05,
            ),
            data_uri=f"s3://ghgsat/scan/{i+1:03d}.nc",
            quality_flag=rng.uniform(0.7, 0.98),
            plume_detected=True,
            plume_center=Coordinate(lon=lon, lat=lat),
            emission_rate_kg_h=round(rate, 1),
            emission_rate_uncertainty_kg_h=round(rate * rng.uniform(0.1, 0.2), 1),
        ))

        # ~60% have Tanager-1 coverage
        if rng.random() < 0.6:
            tanager_obs.append(Tanager1Observation(
                observation_id=f"TAN-SCAN-{i+1:03d}",
                acquisition_time=NOW + timedelta(hours=rng.uniform(0, 6)),
                processing_level=ProcessingLevel.L2,
                bounding_box=BoundingBox(
                    min_lon=lon - 0.05, min_lat=lat - 0.05,
                    max_lon=lon + 0.05, max_lat=lat + 0.05,
                ),
                data_uri=f"s3://tanager/scan/{i+1:03d}.nc",
                quality_flag=rng.uniform(0.8, 0.95),
                methane_band_depth_1650nm=0.01 + rate / 50000,
                methane_band_depth_2300nm=0.008 + rate / 60000,
                swir_snr=rng.uniform(150, 300),
            ))

    region = BoundingBox(min_lon=min_lon, min_lat=min_lat,
                         max_lon=max_lon, max_lat=max_lat)
    methanesat_obs = [MethaneSATObservation(
        observation_id="MSAT-SCAN-001",
        acquisition_time=NOW + timedelta(hours=4),
        processing_level=ProcessingLevel.L2,
        bounding_box=region,
        data_uri="s3://methanesat/scan/001.nc",
        quality_flag=0.88,
        area_flux_kg_km2_h=8.5,
        enhancement_ppb=35.0,
        background_ch4_ppb=1900.0,
    )]

    s5p_obs = [
        Sentinel5PObservation(
            observation_id=f"S5P-SCAN-{d:03d}",
            acquisition_time=NOW - timedelta(days=d),
            processing_level=ProcessingLevel.L2,
            bounding_box=region,
            data_uri=f"s3://s5p/scan/{d:03d}.nc",
            quality_flag=0.7,
            xch4_ppb=1910 + rng.uniform(0, 50),
            qa_value=rng.uniform(0.6, 0.85),
            cloud_fraction=rng.uniform(0.0, 0.2),
        )
        for d in range(1, 22)
    ]

    return ghgsat_obs, tanager_obs, methanesat_obs, s5p_obs


# ── Routes ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("viewer.html")


@app.route("/api/scan", methods=["POST"])
def scan():
    """Scan a region for methane emitters and return prioritized results."""
    body = request.get_json(force=True)
    bbox = body.get("bbox")
    min_rate = body.get("min_rate", 0)
    n_emitters = body.get("n_emitters", 30)

    if not bbox or len(bbox) != 4:
        return jsonify({"error": "bbox must be [west, south, east, north]"}), 400

    # Generate synthetic data for the region
    ghgsat, tanager, methanesat, s5p = generate_basin_data(
        bbox, n_emitters=n_emitters
    )

    scanner = EmitterScanner()
    scanner.add_ghgsat_observations(ghgsat)
    scanner.add_tanager_observations(tanager)
    scanner.add_methanesat_observations(methanesat)
    scanner.add_sentinel5p_observations(s5p)

    region = BoundingBox(
        min_lon=bbox[0], min_lat=bbox[1],
        max_lon=bbox[2], max_lat=bbox[3],
    )
    results = scanner.scan(region, min_emission_rate_kg_h=min_rate)
    results.prioritize()

    # Build response
    features = []
    for target in results.targets:
        det = target.detection
        s = target.summary()
        features.append({
            "id": det.detection_id,
            "lon": det.location.lon,
            "lat": det.location.lat,
            "rate_kg_h": det.emission_rate_kg_h,
            "uncertainty_kg_h": det.emission_rate_uncertainty_kg_h,
            "annualized_tonnes": s["annualized_tonnes_ch4"],
            "co2e_tonnes": s["co2e_tonnes_gwp20"],
            "corroboration": s["corroboration"],
            "sensors": s["sensors"],
            "confidence": s["confidence"],
            "persistence_days": s["persistence_days"],
            "priority": s["priority"],
            "priority_score": s["priority_score"],
            "scores": s["score_breakdown"],
            "spectral_confirmed": det.spectral_confirmed,
            "area_flux": det.area_flux_kg_km2_h,
        })

    return jsonify({
        "count": len(features),
        "features": features,
        "stats": results.stats(),
    })


@app.route("/api/plume/<detection_id>.png")
def plume_image(detection_id):
    """Generate a plume visualization for a specific detection."""
    lon = float(request.args.get("lon", -103.72))
    lat = float(request.args.get("lat", 31.81))
    rate = float(request.args.get("rate", 500))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    extent = 800
    res = 2
    xs = np.arange(-extent, extent, res)
    ys = np.arange(-extent, extent, res)
    X, Y = np.meshgrid(xs, ys)

    # Plume
    import math
    angle = math.radians(40)
    sx, sy = -80, -40
    dx, dy = X - sx, Y - sy
    along = dx * math.cos(angle) + dy * math.sin(angle)
    cross = -dx * math.sin(angle) + dy * math.cos(angle)
    strength = min(rate / 500, 3.0)
    plume = np.where(
        along > 0,
        strength * np.exp(-0.5 * (cross / 90) ** 2) * np.exp(-0.5 * ((along - 150) / 280) ** 2),
        strength * 0.3 * np.exp(-0.5 * ((dx**2 + dy**2) / 60**2)),
    )

    # Terrain
    rng = np.random.RandomState(int(abs(lon * 1000 + lat * 1000)) % 2**31)
    terrain = rng.normal(0.5, 0.08, X.shape)
    from scipy.ndimage import gaussian_filter
    terrain = gaussian_filter(terrain, sigma=35)
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())

    terrain_cmap = LinearSegmentedColormap.from_list(
        "t", ["#c2b280", "#b8a97a", "#d4c8a0", "#c9bc8e", "#bfaf7a"])
    plume_cmap = LinearSegmentedColormap.from_list(
        "p", [(0,0,0,0), (1,0.85,0,0.15), (1,0.5,0,0.4), (1,0.15,0,0.7), (0.7,0,0.1,0.9)])

    ax.imshow(terrain, extent=[-extent, extent, -extent, extent],
              cmap=terrain_cmap, alpha=0.5, origin="lower")
    ax.imshow(plume, extent=[-extent, extent, -extent, extent],
              cmap=plume_cmap, origin="lower", vmin=0, vmax=0.8)

    # Contours
    levels = [0.15, 0.3, 0.5]
    ax.contour(X, Y, plume, levels=levels,
               colors=["#00ff88", "#ffaa00", "#ff2200"],
               linewidths=0.8, alpha=0.7)

    # Source
    ax.plot(sx, sy, "x", color="#ff0044", markersize=14, markeredgewidth=3, zorder=10)

    # Info box
    info = (
        f"{rate:,.0f} kg CH\u2084/hr\n"
        f"{lat:.4f}\u00b0N, {abs(lon):.4f}\u00b0W\n"
        f"{detection_id}"
    )
    ax.text(extent - 20, -extent + 20, info,
            fontsize=8, fontfamily="monospace", color="#00ff88",
            va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d1117",
                      edgecolor="#00ff88", alpha=0.9),
            path_effects=[pe.withStroke(linewidth=1, foreground="#0d1117")],
            zorder=12)

    # Scale bar
    ax.plot([-extent+30, -extent+230], [-extent+30, -extent+30],
            color="white", linewidth=2, zorder=15)
    ax.text(-extent+130, -extent+55, "200 m",
            ha="center", fontsize=7, color="white", zorder=15)

    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return Response(buf.read(), content_type="image/png")


if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser(description="Methane Prosecution Viewer")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5001)))
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print(f"\n  Methane Prosecution Viewer at http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=False)
