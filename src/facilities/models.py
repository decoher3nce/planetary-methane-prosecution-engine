"""
Facility data models for infrastructure identification and attribution.

These models represent the physical facilities that can be identified
in Sentinel-2 optical imagery and linked to methane emission sources.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from src.sensors.models import BoundingBox, Coordinate


class FacilityType(str, Enum):
    """Types of methane-emitting facilities identifiable in optical imagery."""

    WELL_PAD = "well_pad"
    COMPRESSOR_STATION = "compressor_station"
    PROCESSING_PLANT = "processing_plant"
    PIPELINE = "pipeline"
    PIPELINE_JUNCTION = "pipeline_junction"
    TANK_BATTERY = "tank_battery"
    FLARE_STACK = "flare_stack"
    LANDFILL = "landfill"
    WASTEWATER_LAGOON = "wastewater_lagoon"
    COAL_MINE = "coal_mine"
    FEEDLOT = "feedlot"
    UNKNOWN = "unknown"


class InfrastructureSignature(str, Enum):
    """Optical/spectral signatures used to identify facility types in Sentinel-2."""

    CLEARED_PAD = "cleared_pad"  # Bare earth rectangle (well pad)
    CIRCULAR_TANK = "circular_tank"  # Storage/holding tanks
    LINEAR_CORRIDOR = "linear_corridor"  # Pipeline right-of-way
    THERMAL_ANOMALY = "thermal_anomaly"  # Flare stack / compressor heat
    IMPOUNDMENT = "impoundment"  # Wastewater or waste containment
    DISTURBED_SOIL = "disturbed_soil"  # Recent construction/activity
    ROAD_ACCESS = "road_access"  # Access roads to remote sites
    VEGETATION_STRESS = "vegetation_stress"  # NDVI anomaly from emissions


class FacilityFootprint(BaseModel):
    """Physical footprint of an identified facility."""

    centroid: Coordinate
    bounding_box: BoundingBox
    area_m2: float = Field(ge=0, description="Estimated facility area in m²")
    perimeter_m: float = Field(ge=0, description="Estimated perimeter in meters")
    orientation_deg: Optional[float] = Field(
        default=None, ge=0, lt=360,
        description="Major axis orientation in degrees from north",
    )


class Facility(BaseModel):
    """An identified facility linked to a methane emission source.

    Represents a physical piece of infrastructure identified through
    Sentinel-2 optical imagery analysis that is a candidate source
    for an observed methane emission.
    """

    facility_id: str
    facility_type: FacilityType
    name: Optional[str] = None
    operator: Optional[str] = None
    footprint: FacilityFootprint
    signatures_detected: List[InfrastructureSignature] = Field(default_factory=list)

    # Identification confidence
    identification_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in facility type classification (0-1)",
    )

    # Sentinel-2 source data
    sentinel2_observation_id: Optional[str] = None
    sentinel2_acquisition_time: Optional[datetime] = None

    # Spectral indices from Sentinel-2
    ndvi_mean: Optional[float] = Field(
        default=None, description="Mean NDVI over facility footprint (-1 to 1)",
    )
    ndbi_mean: Optional[float] = Field(
        default=None, description="Mean NDBI (built-up index) over footprint",
    )

    # Regulatory identifiers (when matched to databases)
    epa_ghgrp_id: Optional[str] = Field(
        default=None, description="EPA GHGRP facility ID",
    )
    state_permit_id: Optional[str] = None
    api_well_number: Optional[str] = Field(
        default=None, description="API well number (14-digit)",
    )

    # Attribution metadata
    distance_to_plume_m: Optional[float] = Field(
        default=None, description="Distance from facility centroid to plume center",
    )
    upwind_of_plume: Optional[bool] = Field(
        default=None, description="Whether facility is upwind of the detected plume",
    )

    def to_attribution_dict(self) -> dict:
        """Serialize for attribution reports."""
        return {
            "facility_id": self.facility_id,
            "facility_type": self.facility_type.value,
            "name": self.name,
            "operator": self.operator,
            "location": {
                "lon": self.footprint.centroid.lon,
                "lat": self.footprint.centroid.lat,
            },
            "area_m2": self.footprint.area_m2,
            "identification_confidence": self.identification_confidence,
            "signatures": [s.value for s in self.signatures_detected],
            "distance_to_plume_m": self.distance_to_plume_m,
            "upwind_of_plume": self.upwind_of_plume,
            "regulatory_ids": {
                "epa_ghgrp": self.epa_ghgrp_id,
                "state_permit": self.state_permit_id,
                "api_well": self.api_well_number,
            },
        }
