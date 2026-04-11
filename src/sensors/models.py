"""
Data models for each sensor in the methane detection stack.

Each model captures the essential observational parameters plus metadata
required for chain-of-custody: acquisition time, geolocation, processing
level, and instrument-specific quality indicators.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, Field, field_validator


class SensorType(str, Enum):
    """Sensors in the prosecution stack."""

    TANAGER_1 = "planet_tanager_1"
    GHGSAT = "ghgsat"
    METHANESAT = "methanesat"
    SENTINEL_5P = "sentinel_5p_tropomi"
    SENTINEL_2 = "sentinel_2"


class ProcessingLevel(str, Enum):
    """Standard remote-sensing processing levels."""

    L0 = "L0"  # Raw instrument data
    L1A = "L1A"  # Reconstructed, unprocessed
    L1B = "L1B"  # Radiometrically calibrated
    L2 = "L2"  # Geophysical variable (e.g., CH4 column)
    L3 = "L3"  # Gridded / composited
    L4 = "L4"  # Model output / assimilated


class BoundingBox(BaseModel):
    """WGS-84 bounding box."""

    min_lon: float = Field(ge=-180, le=180)
    min_lat: float = Field(ge=-90, le=90)
    max_lon: float = Field(ge=-180, le=180)
    max_lat: float = Field(ge=-90, le=90)

    @field_validator("max_lon")
    @classmethod
    def lon_order(cls, v: float, info) -> float:
        if "min_lon" in info.data and v < info.data["min_lon"]:
            raise ValueError("max_lon must be >= min_lon")
        return v

    @field_validator("max_lat")
    @classmethod
    def lat_order(cls, v: float, info) -> float:
        if "min_lat" in info.data and v < info.data["min_lat"]:
            raise ValueError("max_lat must be >= min_lat")
        return v


class Coordinate(BaseModel):
    """WGS-84 point coordinate."""

    lon: float = Field(ge=-180, le=180)
    lat: float = Field(ge=-90, le=90)


class SensorObservation(BaseModel):
    """Base class for all sensor observations.

    Every observation carries a content hash computed at ingest time.
    This hash anchors the chain-of-custody — any modification to the
    observation data will produce a different hash, breaking the chain.
    """

    sensor: SensorType
    observation_id: str
    acquisition_time: datetime
    processing_level: ProcessingLevel
    bounding_box: BoundingBox
    data_provider: str
    data_uri: str  # Where the raw data lives (S3, GCS, local path)
    quality_flag: float = Field(ge=0.0, le=1.0, description="0=unusable, 1=perfect")
    content_hash: Optional[str] = None  # SHA-256 of the raw data payload

    def compute_content_hash(self, raw_bytes: bytes) -> str:
        """Compute and store SHA-256 hash of raw observation data."""
        self.content_hash = hashlib.sha256(raw_bytes).hexdigest()
        return self.content_hash

    def to_evidence_dict(self) -> dict:
        """Serialize to a dictionary suitable for evidence chain records."""
        d = self.model_dump(mode="json")
        d["acquisition_time"] = self.acquisition_time.isoformat()
        return d


# ── Sensor-Specific Observation Models ──────────────────────────────


class Tanager1Observation(SensorObservation):
    """Planet Tanager-1: 400-band VSWIR hyperspectral (400–2500 nm).

    Key capability: Full spectral fingerprinting of methane absorption
    features at 1.65 µm and 2.3 µm bands with ~30m spatial resolution.
    """

    sensor: SensorType = SensorType.TANAGER_1
    data_provider: str = "Planet Labs"
    num_bands: int = Field(default=400, description="Number of spectral bands")
    spatial_resolution_m: float = Field(default=30.0, description="GSD in meters")
    spectral_range_nm: tuple = (400.0, 2500.0)
    methane_band_depth_1650nm: Optional[float] = Field(
        default=None, description="Absorption depth at 1.65 µm CH4 feature"
    )
    methane_band_depth_2300nm: Optional[float] = Field(
        default=None, description="Absorption depth at 2.3 µm CH4 feature"
    )
    swir_snr: Optional[float] = Field(
        default=None, description="SWIR signal-to-noise ratio"
    )


class GHGSatObservation(SensorObservation):
    """GHGSat: High-resolution (25m) point-source methane detection.

    Key capability: Pinpoints individual emission sources with enough
    spatial precision to identify specific equipment (e.g., a single wellhead).
    """

    sensor: SensorType = SensorType.GHGSAT
    data_provider: str = "GHGSat Inc."
    spatial_resolution_m: float = Field(default=25.0, description="GSD in meters")
    emission_rate_kg_h: Optional[float] = Field(
        default=None, description="Estimated point-source emission rate (kg CH4/hr)"
    )
    emission_rate_uncertainty_kg_h: Optional[float] = Field(
        default=None, description="1-sigma uncertainty on emission rate"
    )
    plume_detected: bool = False
    plume_center: Optional[Coordinate] = None
    detection_limit_kg_h: float = Field(
        default=100.0, description="Minimum detectable emission rate"
    )


class MethaneSATObservation(SensorObservation):
    """MethaneSAT: Wide-area methane quantification at basin scale.

    Key capability: Quantifies total methane emissions over large areas
    (200km swath) including diffuse sources that point-source sensors miss.
    """

    sensor: SensorType = SensorType.METHANESAT
    data_provider: str = "MethaneSAT LLC (EDF)"
    spatial_resolution_m: float = Field(default=100.0, description="GSD in meters")
    swath_width_km: float = Field(default=200.0, description="Swath width in km")
    area_flux_kg_km2_h: Optional[float] = Field(
        default=None, description="Area-integrated emission flux (kg CH4/km²/hr)"
    )
    area_flux_uncertainty: Optional[float] = None
    enhancement_ppb: Optional[float] = Field(
        default=None, description="CH4 column enhancement above background (ppb)"
    )
    background_ch4_ppb: Optional[float] = Field(
        default=None, description="Background CH4 mixing ratio (ppb)"
    )


class Sentinel5PObservation(SensorObservation):
    """Sentinel-5P TROPOMI: Global daily methane mapping.

    Key capability: Provides temporal context — daily global coverage
    establishes emission persistence patterns and pre/post baselines.
    """

    sensor: SensorType = SensorType.SENTINEL_5P
    data_provider: str = "ESA Copernicus"
    spatial_resolution_m: float = Field(default=5500.0, description="~5.5km x 7km")
    xch4_ppb: Optional[float] = Field(
        default=None, description="Column-averaged dry-air CH4 mixing ratio (ppb)"
    )
    xch4_precision_ppb: Optional[float] = None
    qa_value: float = Field(
        default=0.5, ge=0.0, le=1.0, description="TROPOMI QA value (>=0.5 recommended)"
    )
    orbit_number: Optional[int] = None
    cloud_fraction: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class Sentinel2Observation(SensorObservation):
    """Sentinel-2 A/B: High-resolution optical for facility identification.

    Key capability: 10m visible/NIR imagery identifies physical
    infrastructure — well pads, compressor stations, landfills, pipelines.
    """

    sensor: SensorType = SensorType.SENTINEL_2
    data_provider: str = "ESA Copernicus"
    spatial_resolution_m: float = Field(default=10.0, description="GSD in meters (B2-B4)")
    cloud_cover_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    tile_id: Optional[str] = None
    relative_orbit: Optional[int] = None


# ── Registry for lookup ─────────────────────────────────────────────

SENSOR_MODEL_REGISTRY: Dict[SensorType, Type[SensorObservation]] = {
    SensorType.TANAGER_1: Tanager1Observation,
    SensorType.GHGSAT: GHGSatObservation,
    SensorType.METHANESAT: MethaneSATObservation,
    SensorType.SENTINEL_5P: Sentinel5PObservation,
    SensorType.SENTINEL_2: Sentinel2Observation,
}
