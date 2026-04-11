# 🚫 Planetary Methane Prosecution Engine

> **⚠️ PROJECT ABANDONED** — This project was an experimental prototype exploring the intersection of remote sensing and climate litigation. It was abandoned because the critical methane detection sensors (GHGSat, Planet Tanager-1, MethaneSAT) require paid commercial access, and the free alternatives (Sentinel-5P at 5.5km resolution) lack the spatial precision needed for facility-level prosecution. The code and architecture are published here for anyone who wants to pick up where we left off.

**Turn spectral signatures into legal evidence** — fuse multi-sensor methane detection into court-ready prosecution packages.

## Concept

No one is currently building the bridge from spectral detection to prosecutable evidence at global scale. This engine was designed to:

1. **Fuse** Tanager-1's 400-band hyperspectral with GHGSat's 25m point-source methane detection and MethaneSAT's area-wide quantification into a single corroborated detection
2. **Identify** responsible facilities using Sentinel-2 10m optical imagery
3. **Generate** automated attribution reports linking specific well pads, landfills, and pipelines to quantified emissions
4. **Package** everything as court-ready evidence with a tamper-evident cryptographic chain of custody

The estimated 100,000+ super-emitter sites are responsible for ~50% of anthropogenic methane. Forcing their closure through litigation could achieve more near-term climate impact than any renewable energy deployment.

## What Was Built

**92 tests passing.** The full pipeline works end-to-end on synthetic data.

```
src/
├── sensors/models.py              # 5 sensor data models (Tanager-1, GHGSat, MethaneSAT, S5P, S2)
├── fusion/engine.py               # Multi-sensor fusion with spatial/temporal matching
├── evidence/
│   ├── chain_of_custody.py        # SHA-256 cryptographic chain-of-custody
│   └── packager.py                # Court-ready evidence packaging with tamper detection
├── facilities/
│   ├── models.py                  # 11 facility types, 8 optical signatures
│   └── identifier.py              # Sentinel-2 facility classification + wind attribution
├── reports/attribution.py         # Attribution reports with regulatory assessment (EPA violations, penalties)
├── scanning/scanner.py            # Area-wide emitter discovery + prosecution prioritization
└── visualization/render_fusion.py # Multi-sensor plume visualization
```

Plus a **Flask web viewer** (`viewer.py`) with Leaflet map, ranked prosecution queue, and per-emitter plume visualizations.

### Sensor Fusion Stack

| Sensor | Resolution | Role | Access |
|--------|-----------|------|--------|
| Planet Tanager-1 | 30m, 400 bands | Spectral CH₄ confirmation at 1.65µm + 2.3µm | **Paid** (Planet API) |
| GHGSat | 25m | Point-source emission rate quantification | **Paid** (commercial) |
| MethaneSAT | 100m, 200km swath | Basin-wide area flux | **Free** (coming soon) |
| Sentinel-5P TROPOMI | 5.5km | Daily global CH₄ columns, temporal persistence | **Free** (Copernicus) |
| Sentinel-2 A/B | 10m | Facility identification via optical/SWIR | **Free** (Copernicus) |

### Chain of Custody

Every processing step — from raw sensor ingest through evidence packaging — is recorded as a cryptographically linked `ChainLink` with SHA-256 hashes. Tampering with any observation, fusion result, or report breaks the chain, detectable via `chain.verify_integrity()`. Modeled after NIST SP 800-86 digital forensics standards.

### Regulatory Assessment

Automatically evaluates emissions against:
- EPA GHGRP Subpart W (petroleum & natural gas)
- EPA OOOOa/b/c methane standards
- Clean Air Act Section 111
- EPA Super-Emitter Response Program (≥100 kg/hr threshold)
- Estimates civil penalties at $65,000/day/violation

## Why It Was Abandoned

**The sensor data access problem.** The architecture assumes you can fuse observations from multiple methane-specific sensors. In practice:

- **GHGSat** — commercial API, requires paid subscription
- **Planet Tanager-1** — commercial API, requires paid subscription
- **MethaneSAT** — data portal not yet publicly available
- **Sentinel-5P** — free but 5.5km resolution is too coarse for facility-level attribution
- **Sentinel-2** — free but **cannot detect methane** (optical sensor, not a gas sensor)

Without at least one high-resolution methane sensor providing real emission rates, the prosecution engine has no prosecutable evidence to work with. The fusion logic, chain of custody, facility identification, and evidence packaging are all functional — they just need real data.

## If You Want to Continue This

The most viable path:

1. **Get GHGSat access** — contact sales@ghgsat.com, or apply for research access
2. **Get Planet access** — apply to their [Education & Research Program](https://www.planet.com/markets/education-and-research/) for free Tanager-1 data
3. **Wire up Sentinel-5P** — use Copernicus Data Space API for regional CH₄ scanning (free, available now)
4. **Wire up Sentinel-2** — use for facility identification at detected hotspots (free, available now)
5. **Build data connectors** — replace the synthetic data generation in `viewer.py` with real API calls

The engine, tests, and web viewer are all ready for real data.

## Running It

```bash
# Install
pip install -e ".[dev]"

# Run tests (92 passing)
pytest tests/ -v

# Run the single-emitter prosecution example
python3 example_prosecution.py

# Run the basin-wide scan example
python3 example_scan.py

# Run the web viewer
python3 viewer.py --port 5002
```

## Sources

Google Earth Engine Catalog · ESA Copernicus · NASA EarthData · Planet Labs · GHGSat · MethaneSAT (EDF) · EPA GHGRP · WEF Global Risks Report 2026

---

*From the [Spectral Solutions Atlas](https://github.com/decoher3nce) — mapping 80+ geospatial and spectral data sources against 20 critical global problems to surface creative solutions at their intersection.*
