"""
Synthetic multi-sensor methane plume visualization.

Renders a realistic spatial view of how GHGSat (25m), Tanager-1 (30m),
and MethaneSAT (100m) observations align around a methane super-emitter,
showing each sensor's contribution to the prosecution evidence stack.
"""

from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def _gaussian_plume(
    x: np.ndarray,
    y: np.ndarray,
    source_x: float,
    source_y: float,
    wind_angle_deg: float = 45.0,
    emission_strength: float = 1.0,
    sigma_along: float = 300.0,
    sigma_cross: float = 120.0,
) -> np.ndarray:
    """Generate a Gaussian plume shape on a 2D grid (meters)."""
    angle = math.radians(wind_angle_deg)
    dx = x - source_x
    dy = y - source_y
    along = dx * math.cos(angle) + dy * math.sin(angle)
    cross = -dx * math.sin(angle) + dy * math.cos(angle)
    # Only downwind
    plume = np.where(
        along > 0,
        emission_strength
        * np.exp(-0.5 * (cross / sigma_cross) ** 2)
        * np.exp(-0.5 * ((along - 200) / sigma_along) ** 2),
        emission_strength * 0.3 * np.exp(-0.5 * ((dx**2 + dy**2) / 80**2)),
    )
    return plume


def _add_facility_markers(ax, source_x, source_y):
    """Draw well pad infrastructure at the emission source."""
    # Well pad footprint
    pad = mpatches.FancyBboxPatch(
        (source_x - 40, source_y - 30), 80, 60,
        boxstyle="round,pad=5", facecolor="#8B7355", edgecolor="#4a3c28",
        linewidth=1.5, alpha=0.85, zorder=5,
    )
    ax.add_patch(pad)

    # Storage tanks
    for dx, dy, r in [(-20, 10, 12), (15, 10, 10), (25, -8, 8)]:
        circle = plt.Circle(
            (source_x + dx, source_y + dy), r,
            facecolor="#a0a0a0", edgecolor="#555", linewidth=1, alpha=0.9, zorder=6,
        )
        ax.add_patch(circle)

    # Flare stack
    ax.plot(source_x + 35, source_y + 20, marker="^", color="#ff6600",
            markersize=8, zorder=7)

    ax.text(
        source_x, source_y - 50, "WELL PAD WP-4117",
        ha="center", va="top", fontsize=7, fontweight="bold",
        color="#4a3c28", zorder=8,
        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
    )


def render_fusion_visualization(output_path: str = "fusion_visualization.png") -> str:
    """Render the full multi-sensor fusion visualization."""

    # ── Scene setup ──────────────────────────────────────────────
    # 2km x 2km area centered on a Permian Basin well pad
    extent_m = 1000  # +/- from center
    resolution = 2  # meters per pixel
    source_x, source_y = -100, -50  # Emission source offset from center

    xs = np.arange(-extent_m, extent_m, resolution)
    ys = np.arange(-extent_m, extent_m, resolution)
    X, Y = np.meshgrid(xs, ys)

    # Generate methane plume (wind from SW)
    plume = _gaussian_plume(X, Y, source_x, source_y,
                            wind_angle_deg=40, emission_strength=1.0,
                            sigma_along=350, sigma_cross=100)

    # ── Colormaps ────────────────────────────────────────────────
    plume_cmap = LinearSegmentedColormap.from_list(
        "ch4_plume",
        [(0, 0, 0, 0), (1, 0.85, 0, 0.15), (1, 0.5, 0, 0.4),
         (1, 0.15, 0, 0.7), (0.7, 0, 0.1, 0.9)],
    )

    terrain_cmap = LinearSegmentedColormap.from_list(
        "arid_terrain",
        ["#c2b280", "#b8a97a", "#d4c8a0", "#c9bc8e", "#bfaf7a"],
    )

    # Synthetic terrain (arid, flat with subtle variation)
    np.random.seed(42)
    terrain = np.random.normal(0.5, 0.08, X.shape)
    # Low-freq variation
    from scipy.ndimage import gaussian_filter
    terrain = gaussian_filter(terrain, sigma=40)
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())

    # ── Figure with 4 panels ─────────────────────────────────────
    fig = plt.figure(figsize=(18, 16), facecolor="#1a1a2e")
    fig.suptitle(
        "PLANETARY METHANE PROSECUTION ENGINE\nMulti-Sensor Fusion — Permian Basin Super-Emitter",
        fontsize=16, fontweight="bold", color="white", y=0.97,
    )

    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22,
                          left=0.06, right=0.94, top=0.91, bottom=0.06)

    panels = [
        (gs[0, 0], "GHGSat — 25m Point-Source Detection",
         "ghgsat", 25),
        (gs[0, 1], "Planet Tanager-1 — 400-Band Hyperspectral Confirmation",
         "tanager", 30),
        (gs[1, 0], "MethaneSAT — Area-Wide Quantification (200km swath)",
         "methanesat", 100),
        (gs[1, 1], "FUSED DETECTION — Prosecution-Grade Evidence",
         "fused", None),
    ]

    for gs_pos, title, sensor, res in panels:
        ax = fig.add_subplot(gs_pos)
        ax.set_facecolor("#0d1117")

        if sensor == "ghgsat":
            _render_ghgsat(ax, X, Y, plume, terrain, terrain_cmap, plume_cmap,
                           source_x, source_y, extent_m)
        elif sensor == "tanager":
            _render_tanager(ax, X, Y, plume, terrain, terrain_cmap,
                            source_x, source_y, extent_m)
        elif sensor == "methanesat":
            _render_methanesat(ax, X, Y, plume, terrain, terrain_cmap, plume_cmap,
                               source_x, source_y, extent_m)
        elif sensor == "fused":
            _render_fused(ax, X, Y, plume, terrain, terrain_cmap, plume_cmap,
                          source_x, source_y, extent_m)

        ax.set_title(title, fontsize=10, fontweight="bold", color="white", pad=8)
        ax.set_xlim(-extent_m, extent_m)
        ax.set_ylim(-extent_m, extent_m)
        ax.set_aspect("equal")
        ax.tick_params(colors="#666", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#333")

    # ── Bottom legend bar ────────────────────────────────────────
    legend_text = (
        "Case PMPE-7A3F19C2  •  31.81°N, 103.72°W  •  "
        "Emission Rate: 500 ± 75 kg CH₄/hr  •  "
        "Corroboration: TRIPLE (prosecution-grade)  •  "
        "Confidence: 92.3%  •  Chain of Custody: VERIFIED"
    )
    fig.text(
        0.5, 0.015, legend_text,
        ha="center", va="bottom", fontsize=9, color="#00ff88",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#0d1117",
                  edgecolor="#00ff88", alpha=0.9),
    )

    plt.savefig(output_path, dpi=180, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)
    return output_path


def _render_ghgsat(ax, X, Y, plume, terrain, terrain_cmap, plume_cmap,
                   sx, sy, extent):
    """GHGSat panel: highest-resolution point-source detection."""
    ax.imshow(terrain, extent=[-extent, extent, -extent, extent],
              cmap=terrain_cmap, alpha=0.6, origin="lower")

    # Pixelate to 25m resolution
    block = 25 // 2  # pixels per GHGSat cell
    plume_ghg = _pixelate(plume, block)
    im = ax.imshow(plume_ghg, extent=[-extent, extent, -extent, extent],
                   cmap=plume_cmap, origin="lower", vmin=0, vmax=0.8)

    # Plume contours
    levels = [0.15, 0.3, 0.5, 0.7]
    ax.contour(X, Y, plume_ghg, levels=levels, colors=["#ffaa00", "#ff6600", "#ff2200", "#cc0000"],
               linewidths=0.8, alpha=0.7)

    # Source marker
    ax.plot(sx, sy, "x", color="#ff0044", markersize=12, markeredgewidth=2.5, zorder=10)
    ax.annotate(
        f"500 ± 75 kg/hr", (sx, sy), xytext=(sx + 120, sy + 130),
        fontsize=8, color="#ff4466", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#ff4466", lw=1.5),
        path_effects=[pe.withStroke(linewidth=2, foreground="#0d1117")],
        zorder=11,
    )

    # Resolution grid hint
    _add_resolution_grid(ax, 25, extent, color="#ffaa00")
    _add_sensor_badge(ax, "25m GSD", "#ff4466", extent)
    _add_scalebar(ax, extent)


def _render_tanager(ax, X, Y, plume, terrain, terrain_cmap, sx, sy, extent):
    """Tanager-1 panel: spectral absorption band confirmation."""
    ax.imshow(terrain, extent=[-extent, extent, -extent, extent],
              cmap=terrain_cmap, alpha=0.6, origin="lower")

    # Simulate SWIR absorption depth at 1.65µm — correlates with plume
    absorption = plume * 0.045 / plume.max() if plume.max() > 0 else plume
    block = 30 // 2
    absorption_px = _pixelate(absorption, block)

    # Spectral colormap (blue-green for absorption depth)
    spec_cmap = LinearSegmentedColormap.from_list(
        "swir_absorption",
        [(0, 0, 0, 0), (0, 0.4, 0.8, 0.2), (0, 0.7, 1.0, 0.5),
         (0.2, 1.0, 0.8, 0.7), (0.8, 1.0, 0.3, 0.9)],
    )
    ax.imshow(absorption_px, extent=[-extent, extent, -extent, extent],
              cmap=spec_cmap, origin="lower", vmin=0, vmax=0.04)

    # Spectral profile inset
    inset = ax.inset_axes([0.58, 0.62, 0.38, 0.32])
    inset.set_facecolor("#0d1117")
    wavelengths = np.linspace(1500, 2500, 200)
    # Simulated CH4 absorption spectrum with features at 1650nm and 2300nm
    spectrum = 1.0 - 0.045 * np.exp(-0.5 * ((wavelengths - 1650) / 30) ** 2) \
                    - 0.032 * np.exp(-0.5 * ((wavelengths - 2300) / 40) ** 2) \
                    + np.random.normal(0, 0.003, len(wavelengths))
    inset.plot(wavelengths, spectrum, color="#00ccff", linewidth=1.2)
    inset.axvspan(1630, 1670, alpha=0.3, color="#ff4444", label="CH₄ 1.65µm")
    inset.axvspan(2280, 2320, alpha=0.3, color="#ff8844", label="CH₄ 2.3µm")
    inset.set_xlabel("λ (nm)", fontsize=6, color="#888")
    inset.set_ylabel("Reflectance", fontsize=6, color="#888")
    inset.set_title("SWIR Absorption", fontsize=7, color="#00ccff", pad=2)
    inset.tick_params(labelsize=5, colors="#666")
    for spine in inset.spines.values():
        spine.set_color("#333")
    inset.legend(fontsize=5, loc="lower left", framealpha=0.5,
                 labelcolor="white", facecolor="#0d1117", edgecolor="#333")

    ax.plot(sx, sy, "x", color="#00ccff", markersize=12, markeredgewidth=2.5, zorder=10)
    ax.annotate(
        "CH₄ band depth: 0.045\n(1.65µm confirmed)", (sx, sy),
        xytext=(sx + 100, sy + 150),
        fontsize=7, color="#00ccff", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#00ccff", lw=1.5),
        path_effects=[pe.withStroke(linewidth=2, foreground="#0d1117")],
        zorder=11,
    )

    _add_resolution_grid(ax, 30, extent, color="#00ccff")
    _add_sensor_badge(ax, "30m GSD • 400 bands", "#00ccff", extent)
    _add_scalebar(ax, extent)


def _render_methanesat(ax, X, Y, plume, terrain, terrain_cmap, plume_cmap,
                       sx, sy, extent):
    """MethaneSAT panel: wide-area flux with coarser resolution."""
    ax.imshow(terrain, extent=[-extent, extent, -extent, extent],
              cmap=terrain_cmap, alpha=0.4, origin="lower")

    # Heavily pixelated to simulate 100m resolution
    block = 100 // 2
    plume_msat = _pixelate(plume, block)

    # Area flux colormap
    flux_cmap = LinearSegmentedColormap.from_list(
        "area_flux",
        [(0, 0, 0, 0), (0.2, 0, 0.5, 0.15), (0.6, 0, 0.8, 0.35),
         (1.0, 0.3, 0.9, 0.55), (1.0, 0.7, 0.4, 0.75)],
    )
    ax.imshow(plume_msat, extent=[-extent, extent, -extent, extent],
              cmap=flux_cmap, origin="lower", vmin=0, vmax=0.6)

    # Area flux grid overlay
    _add_resolution_grid(ax, 100, extent, color="#aa44ff")

    # Basin-wide annotation
    ax.annotate(
        "Area flux: 12.5 kg CH₄/km²/hr\nEnhancement: +45 ppb\nBackground: 1900 ppb",
        (sx + 200, sy + 200), fontsize=7, color="#cc88ff", fontweight="bold",
        path_effects=[pe.withStroke(linewidth=2, foreground="#0d1117")],
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d1117",
                  edgecolor="#aa44ff", alpha=0.85),
        zorder=11,
    )

    ax.plot(sx, sy, "x", color="#aa44ff", markersize=12, markeredgewidth=2.5, zorder=10)
    _add_sensor_badge(ax, "100m GSD • 200km swath", "#aa44ff", extent)
    _add_scalebar(ax, extent)


def _render_fused(ax, X, Y, plume, terrain, terrain_cmap, plume_cmap,
                  sx, sy, extent):
    """Fused detection panel: all sensors combined with evidence overlay."""
    ax.imshow(terrain, extent=[-extent, extent, -extent, extent],
              cmap=terrain_cmap, alpha=0.5, origin="lower")

    # Full-resolution plume
    fused_cmap = LinearSegmentedColormap.from_list(
        "fused_evidence",
        [(0, 0, 0, 0), (0, 1, 0.5, 0.1), (1, 0.9, 0, 0.3),
         (1, 0.4, 0, 0.6), (1, 0, 0, 0.85)],
    )
    ax.imshow(plume, extent=[-extent, extent, -extent, extent],
              cmap=fused_cmap, origin="lower", vmin=0, vmax=0.7)

    # Contours from GHGSat resolution
    block = 25 // 2
    plume_ghg = _pixelate(plume, block)
    ax.contour(X, Y, plume_ghg, levels=[0.15, 0.3, 0.5],
               colors=["#00ff88", "#ffaa00", "#ff2200"],
               linewidths=1.0, alpha=0.8)

    # Facility marker
    _add_facility_markers(ax, sx, sy)

    # Sensor contribution rings
    for radius, color, label in [
        (150, "#ff4466", "GHGSat 25m"),
        (200, "#00ccff", "Tanager-1 30m"),
        (400, "#aa44ff", "MethaneSAT 100m"),
    ]:
        circle = plt.Circle(
            (sx, sy), radius, fill=False, edgecolor=color,
            linewidth=1.2, linestyle="--", alpha=0.6, zorder=8,
        )
        ax.add_patch(circle)
        angle = math.radians({"GHGSat 25m": -30, "Tanager-1 30m": 60,
                               "MethaneSAT 100m": 150}[label])
        lx = sx + radius * math.cos(angle)
        ly = sy + radius * math.sin(angle)
        ax.text(lx, ly, label, fontsize=6, color=color, ha="center",
                path_effects=[pe.withStroke(linewidth=2, foreground="#0d1117")],
                zorder=9)

    # Evidence box
    evidence_text = (
        "━━ PROSECUTION EVIDENCE ━━\n"
        "Corroboration: TRIPLE\n"
        "GHGSat:     500 ± 75 kg/hr  ✓\n"
        "Tanager-1:  CH₄ 1.65µm 0.045  ✓\n"
        "MethaneSAT: 12.5 kg/km²/hr  ✓\n"
        "Chain of custody: INTACT"
    )
    ax.text(
        extent - 30, -extent + 30, evidence_text,
        fontsize=7, fontfamily="monospace", color="#00ff88",
        va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#0d1117",
                  edgecolor="#00ff88", alpha=0.9),
        path_effects=[pe.withStroke(linewidth=1, foreground="#0d1117")],
        zorder=12,
    )

    _add_sensor_badge(ax, "FUSED • 3 SENSORS", "#00ff88", extent)
    _add_scalebar(ax, extent)


def _pixelate(data: np.ndarray, block_size: int) -> np.ndarray:
    """Reduce resolution by averaging blocks, then upscale back to original shape."""
    if block_size <= 1:
        return data.copy()
    h, w = data.shape
    bh = (h // block_size) * block_size
    bw = (w // block_size) * block_size
    trimmed = data[:bh, :bw]
    reduced = trimmed.reshape(bh // block_size, block_size,
                              bw // block_size, block_size).mean(axis=(1, 3))
    upscaled = np.repeat(np.repeat(reduced, block_size, axis=0),
                         block_size, axis=1)
    # Pad back to original shape if needed
    result = np.zeros_like(data)
    result[:upscaled.shape[0], :upscaled.shape[1]] = upscaled
    return result


def _add_resolution_grid(ax, res_m: int, extent: int, color: str):
    """Draw a faint grid showing sensor pixel boundaries."""
    step = res_m
    for x in range(-extent, extent, step):
        ax.axvline(x, color=color, linewidth=0.15, alpha=0.3)
    for y in range(-extent, extent, step):
        ax.axhline(y, color=color, linewidth=0.15, alpha=0.3)


def _add_sensor_badge(ax, text: str, color: str, extent: int):
    """Add a sensor ID badge in the top-left."""
    ax.text(
        -extent + 20, extent - 20, text,
        fontsize=7, fontweight="bold", color=color,
        va="top", ha="left", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d1117",
                  edgecolor=color, alpha=0.85),
        zorder=15,
    )


def _add_scalebar(ax, extent: int):
    """Add a 200m scale bar."""
    bar_y = -extent + 40
    bar_x = -extent + 40
    bar_len = 200
    ax.plot([bar_x, bar_x + bar_len], [bar_y, bar_y],
            color="white", linewidth=2.5, zorder=15)
    ax.text(bar_x + bar_len / 2, bar_y + 25, "200 m",
            ha="center", fontsize=6, color="white", zorder=15)


if __name__ == "__main__":
    path = render_fusion_visualization()
    print(f"Saved to {path}")
