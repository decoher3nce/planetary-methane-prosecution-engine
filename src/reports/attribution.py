"""
Automated Attribution Report Generator.

Produces structured, machine-readable attribution reports that link
specific facilities to quantified methane emissions. Each report
combines:

  - Multi-sensor fused detection data (Step 1)
  - Sentinel-2 facility identification (Step 2)
  - Emission quantification with uncertainty bounds
  - Chain-of-custody audit trail
  - Regulatory context (EPA thresholds, reporting obligations)

Reports are generated as structured data (JSON/dict) and can be
rendered to human-readable formats via Jinja2 templates in Step 4.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional
from uuid import uuid4

from src.evidence.chain_of_custody import ChainOfCustody, CustodyAction
from src.facilities.models import Facility, FacilityType
from src.fusion.engine import (
    CorroborationLevel,
    FusedDetection,
    FusionConfidence,
)
from src.sensors.models import SensorType


class ViolationType(str, Enum):
    """Categories of regulatory violation that can be evidenced."""

    EPA_SUBPART_W = "epa_subpart_w"  # EPA GHGRP Subpart W (petroleum & gas)
    EPA_OOOO = "epa_oooo"  # EPA OOOOa/b/c methane standards
    CLEAN_AIR_ACT = "clean_air_act"  # Clean Air Act Section 111
    STATE_PERMIT = "state_permit"  # State-level permit violation
    CARBON_CREDIT_FRAUD = "carbon_credit_fraud"  # Fraudulent offset claims
    NEGLIGENT_EMISSION = "negligent_emission"  # Negligent venting/leaking
    SUPER_EMITTER_RESPONSE = "super_emitter_response"  # EPA Super-Emitter Program


class SeverityLevel(str, Enum):
    """Severity classification of the emission violation."""

    MINOR = "minor"  # <100 kg/hr — reportable
    SIGNIFICANT = "significant"  # 100–500 kg/hr — enforcement action
    MAJOR = "major"  # 500–2000 kg/hr — immediate action required
    CRITICAL = "critical"  # >2000 kg/hr — emergency response


@dataclass
class EmissionQuantification:
    """Quantified emission with uncertainty and regulatory context."""

    rate_kg_h: float
    rate_uncertainty_kg_h: float
    rate_tonnes_year: float  # Annualized
    rate_co2e_tonnes_year: float  # CO2-equivalent (GWP-20 or GWP-100)
    gwp_basis: str  # "GWP-20" or "GWP-100"
    quantification_method: str  # Which sensor(s) provided the rate
    detection_limit_kg_h: float

    @staticmethod
    def from_detection(
        detection: FusedDetection,
        gwp_factor: float = 80.0,  # GWP-20 default (IPCC AR6)
        gwp_basis: str = "GWP-20",
    ) -> "EmissionQuantification":
        """Compute emission quantification from a fused detection."""
        rate = detection.emission_rate_kg_h or 0.0
        uncertainty = detection.emission_rate_uncertainty_kg_h or 0.0

        # Annualize: kg/hr → tonnes/year (assuming continuous emission)
        tonnes_year = rate * 8760 / 1000
        co2e = tonnes_year * gwp_factor

        return EmissionQuantification(
            rate_kg_h=rate,
            rate_uncertainty_kg_h=uncertainty,
            rate_tonnes_year=round(tonnes_year, 1),
            rate_co2e_tonnes_year=round(co2e, 1),
            gwp_basis=gwp_basis,
            quantification_method=detection.emission_rate_source or "unknown",
            detection_limit_kg_h=100.0,  # GHGSat nominal detection limit
        )


@dataclass
class RegulatoryContext:
    """Regulatory thresholds and applicable violations."""

    applicable_violations: List[ViolationType] = field(default_factory=list)
    severity: SeverityLevel = SeverityLevel.MINOR
    epa_reporting_threshold_tonnes_yr: float = 25_000  # GHGRP threshold
    exceeds_reporting_threshold: bool = False
    super_emitter_threshold_kg_h: float = 100.0  # EPA Super-Emitter Program
    qualifies_as_super_emitter: bool = False
    estimated_penalty_usd: Optional[float] = None

    @staticmethod
    def assess(
        emission: EmissionQuantification,
        facility_type: FacilityType,
    ) -> "RegulatoryContext":
        """Assess regulatory implications of the emission."""
        ctx = RegulatoryContext()

        rate = emission.rate_kg_h

        # Severity classification
        if rate >= 2000:
            ctx.severity = SeverityLevel.CRITICAL
        elif rate >= 500:
            ctx.severity = SeverityLevel.MAJOR
        elif rate >= 100:
            ctx.severity = SeverityLevel.SIGNIFICANT
        else:
            ctx.severity = SeverityLevel.MINOR

        # Super-emitter qualification
        if rate >= ctx.super_emitter_threshold_kg_h:
            ctx.qualifies_as_super_emitter = True
            ctx.applicable_violations.append(ViolationType.SUPER_EMITTER_RESPONSE)

        # EPA reporting threshold
        if emission.rate_co2e_tonnes_year >= ctx.epa_reporting_threshold_tonnes_yr:
            ctx.exceeds_reporting_threshold = True

        # Facility-type-specific violations
        oil_gas_types = {
            FacilityType.WELL_PAD, FacilityType.COMPRESSOR_STATION,
            FacilityType.PROCESSING_PLANT, FacilityType.PIPELINE,
            FacilityType.PIPELINE_JUNCTION, FacilityType.TANK_BATTERY,
            FacilityType.FLARE_STACK,
        }
        if facility_type in oil_gas_types:
            ctx.applicable_violations.append(ViolationType.EPA_SUBPART_W)
            ctx.applicable_violations.append(ViolationType.EPA_OOOO)
            if rate >= 100:
                ctx.applicable_violations.append(ViolationType.CLEAN_AIR_ACT)

        if facility_type == FacilityType.LANDFILL:
            if rate >= 50:
                ctx.applicable_violations.append(ViolationType.CLEAN_AIR_ACT)

        # Estimated civil penalty: EPA can assess up to ~$65,000/day/violation
        # (adjusted annually for inflation)
        if ctx.severity in (SeverityLevel.MAJOR, SeverityLevel.CRITICAL):
            days_assumed = 30  # Conservative: 1 month of violation
            per_day = 65_000
            n_violations = len(ctx.applicable_violations)
            ctx.estimated_penalty_usd = days_assumed * per_day * max(n_violations, 1)

        return ctx


@dataclass
class AttributionReport:
    """A complete attribution report linking a facility to emissions.

    This is the primary output of Step 3 — a structured report that
    can be rendered for regulatory submission, litigation support,
    or carbon credit invalidation.
    """

    report_id: str
    generated_at: datetime
    case_id: str

    # Detection summary
    detection_id: str
    detection_location: dict  # {lon, lat}
    detection_time: str
    corroboration_level: str
    confidence_score: float
    sensors_used: List[str]

    # Attributed facility
    facility: dict  # Facility.to_attribution_dict()

    # Emission quantification
    emission: dict

    # Regulatory context
    regulatory: dict

    # Chain of custody summary
    chain_of_custody_case_id: str
    chain_integrity_verified: bool
    chain_link_count: int
    audit_trail: List[dict]

    def to_dict(self) -> dict:
        """Full report as a dictionary."""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at.isoformat(),
            "case_id": self.case_id,
            "detection": {
                "detection_id": self.detection_id,
                "location": self.detection_location,
                "time": self.detection_time,
                "corroboration_level": self.corroboration_level,
                "confidence_score": self.confidence_score,
                "sensors_used": self.sensors_used,
            },
            "facility": self.facility,
            "emission": self.emission,
            "regulatory": self.regulatory,
            "chain_of_custody": {
                "case_id": self.chain_of_custody_case_id,
                "integrity_verified": self.chain_integrity_verified,
                "link_count": self.chain_link_count,
                "audit_trail": self.audit_trail,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


class ReportGenerator:
    """Generates automated attribution reports.

    Combines outputs from the fusion engine (Step 1) and facility
    identifier (Step 2) into a complete, structured attribution report.
    """

    def generate(
        self,
        detection: FusedDetection,
        facility: Facility,
        chain: ChainOfCustody,
        gwp_factor: float = 80.0,
        gwp_basis: str = "GWP-20",
    ) -> AttributionReport:
        """Generate an attribution report for a detection-facility pair.

        Args:
            detection: Fused multi-sensor detection from Step 1.
            facility: Identified facility from Step 2.
            chain: Chain of custody tracking all processing steps.
            gwp_factor: Global warming potential multiplier.
            gwp_basis: "GWP-20" or "GWP-100".

        Returns:
            A complete AttributionReport.
        """
        # Quantify emissions
        emission = EmissionQuantification.from_detection(
            detection, gwp_factor=gwp_factor, gwp_basis=gwp_basis
        )

        # Assess regulatory context
        regulatory = RegulatoryContext.assess(emission, facility.facility_type)

        # Record report generation in chain of custody
        chain.add_link(
            action=CustodyAction.REPORT,
            actor="pmpe.reports.generator",
            description=(
                f"Generated attribution report linking {facility.facility_type.value} "
                f"{facility.facility_id} to detection {detection.detection_id} — "
                f"emission rate {emission.rate_kg_h} kg/hr, "
                f"severity: {regulatory.severity.value}"
            ),
            input_hashes=[
                hashlib.sha256(detection.detection_id.encode()).hexdigest(),
                hashlib.sha256(facility.facility_id.encode()).hexdigest(),
            ],
            metadata={
                "detection_id": detection.detection_id,
                "facility_id": facility.facility_id,
                "emission_rate_kg_h": emission.rate_kg_h,
                "severity": regulatory.severity.value,
                "violations": [v.value for v in regulatory.applicable_violations],
            },
        )

        # Verify chain integrity
        chain_valid, chain_errors = chain.verify_integrity()

        report = AttributionReport(
            report_id=f"RPT-{uuid4().hex[:8].upper()}",
            generated_at=datetime.now(timezone.utc),
            case_id=chain.case_id,
            detection_id=detection.detection_id,
            detection_location={
                "lon": detection.location.lon,
                "lat": detection.location.lat,
            },
            detection_time=detection.detection_time.isoformat(),
            corroboration_level=detection.corroboration_level.value,
            confidence_score=detection.confidence_score,
            sensors_used=[s.value for s in detection.sensor_types()],
            facility=facility.to_attribution_dict(),
            emission={
                "rate_kg_h": emission.rate_kg_h,
                "rate_uncertainty_kg_h": emission.rate_uncertainty_kg_h,
                "rate_tonnes_year": emission.rate_tonnes_year,
                "rate_co2e_tonnes_year": emission.rate_co2e_tonnes_year,
                "gwp_basis": emission.gwp_basis,
                "quantification_method": emission.quantification_method,
            },
            regulatory={
                "severity": regulatory.severity.value,
                "applicable_violations": [
                    v.value for v in regulatory.applicable_violations
                ],
                "qualifies_as_super_emitter": regulatory.qualifies_as_super_emitter,
                "exceeds_reporting_threshold": regulatory.exceeds_reporting_threshold,
                "estimated_penalty_usd": regulatory.estimated_penalty_usd,
            },
            chain_of_custody_case_id=chain.case_id,
            chain_integrity_verified=chain_valid,
            chain_link_count=len(chain.links),
            audit_trail=chain.get_audit_trail(),
        )

        return report

    def generate_multi(
        self,
        detection: FusedDetection,
        facilities: List[Facility],
        chain: ChainOfCustody,
        gwp_factor: float = 80.0,
        gwp_basis: str = "GWP-20",
    ) -> List[AttributionReport]:
        """Generate reports for all attributed facilities.

        In cases where multiple facilities may contribute to a single
        plume, each gets its own report. The primary attribution
        (highest confidence, upwind) is listed first.
        """
        reports = []
        for facility in facilities:
            report = self.generate(
                detection, facility, chain,
                gwp_factor=gwp_factor, gwp_basis=gwp_basis,
            )
            reports.append(report)
        return reports
