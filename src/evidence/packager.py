"""
Court-Ready Evidence Packaging System.

Assembles all prosecution evidence — fused detections, facility
identifications, attribution reports, chain-of-custody records, and
supporting data — into a single, cryptographically sealed evidence
package suitable for:

  - Climate litigation (tort claims, public nuisance)
  - Regulatory enforcement (EPA, state agencies)
  - Carbon credit invalidation (offset registries)
  - International compliance (Paris Agreement Art. 13)

The package includes:
  1. Human-readable attribution report (rendered from Jinja2 template)
  2. Machine-readable JSON evidence bundle
  3. Chain-of-custody audit trail
  4. Cryptographic manifest (SHA-256 of every artifact)
  5. Package seal (hash-of-hashes for tamper detection)
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from jinja2 import Environment, BaseLoader

from src.evidence.chain_of_custody import ChainOfCustody, CustodyAction
from src.reports.attribution import AttributionReport


# ── Report template ──────────────────────────────────────────────────

REPORT_TEMPLATE = """\
================================================================================
              PLANETARY METHANE PROSECUTION ENGINE
              EVIDENCE PACKAGE — ATTRIBUTION REPORT
================================================================================

Case ID:        {{ case_id }}
Report ID:      {{ report.report_id }}
Generated:      {{ report.generated_at }}
Classification: CONFIDENTIAL — LEGAL EVIDENCE

================================================================================
1. DETECTION SUMMARY
================================================================================

Detection ID:       {{ report.detection_id }}
Location:           {{ report.detection_location.lat }}°N, {{ report.detection_location.lon | abs }}°W
Detection Time:     {{ report.detection_time }}
Corroboration:      {{ report.corroboration_level | upper }}
Confidence Score:   {{ "%.1f" | format(report.confidence_score * 100) }}%
Sensors Used:       {{ report.sensors_used | join(", ") or "See chain of custody" }}

================================================================================
2. ATTRIBUTED FACILITY
================================================================================

Facility ID:        {{ report.facility.facility_id }}
Type:               {{ report.facility.facility_type | replace("_", " ") | title }}
Name:               {{ report.facility.name or "Unknown" }}
Operator:           {{ report.facility.operator or "Unknown" }}
Location:           {{ report.facility.location.lat }}°N, {{ report.facility.location.lon | abs }}°W
Distance to Plume:  {{ "%.0f" | format(report.facility.distance_to_plume_m or 0) }} m
Upwind of Plume:    {{ "YES" if report.facility.upwind_of_plume else "NO" }}
Area:               {{ "%.0f" | format(report.facility.area_m2) }} m²

Identification Signatures:
{% for sig in report.facility.signatures %}  - {{ sig | replace("_", " ") | title }}
{% endfor %}
Regulatory IDs:
  EPA GHGRP:        {{ report.facility.regulatory_ids.epa_ghgrp or "Not matched" }}
  State Permit:     {{ report.facility.regulatory_ids.state_permit or "Not matched" }}
  API Well #:       {{ report.facility.regulatory_ids.api_well or "Not matched" }}

================================================================================
3. EMISSION QUANTIFICATION
================================================================================

Emission Rate:      {{ "%.0f" | format(report.emission.rate_kg_h) }} ± {{ "%.0f" | format(report.emission.rate_uncertainty_kg_h) }} kg CH₄/hr
Annualized:         {{ "%.1f" | format(report.emission.rate_tonnes_year) }} tonnes CH₄/yr
CO₂ Equivalent:     {{ "{:,.0f}".format(report.emission.rate_co2e_tonnes_year) }} tonnes CO₂e/yr ({{ report.emission.gwp_basis }})
Method:             {{ report.emission.quantification_method }}

================================================================================
4. REGULATORY ASSESSMENT
================================================================================

Severity:               {{ report.regulatory.severity | upper }}
Super-Emitter:          {{ "YES — QUALIFIES" if report.regulatory.qualifies_as_super_emitter else "No" }}
Exceeds GHGRP Threshold: {{ "YES" if report.regulatory.exceeds_reporting_threshold else "No" }}

Applicable Violations:
{% for v in report.regulatory.applicable_violations %}  - {{ v | replace("_", " ") | title }}
{% endfor %}
{% if report.regulatory.estimated_penalty_usd %}Estimated Civil Penalty: ${{ "{:,.0f}".format(report.regulatory.estimated_penalty_usd) }}{% endif %}

================================================================================
5. CHAIN OF CUSTODY
================================================================================

Case ID:            {{ report.chain_of_custody_case_id }}
Integrity:          {{ "VERIFIED — ALL HASHES VALID" if report.chain_integrity_verified else "WARNING — INTEGRITY CHECK FAILED" }}
Total Links:        {{ report.chain_link_count }}

Audit Trail:
{% for step in report.audit_trail %}  {{ step.step }}. [{{ step.timestamp }}] {{ step.action | upper }}
     {{ step.description }}
     Hash: {{ step.link_hash }}  Chain: {{ "VALID" if step.chain_valid else "BROKEN" }}
{% endfor %}
================================================================================
CERTIFICATION

This report was generated by the Planetary Methane Prosecution Engine
(PMPE) using automated multi-sensor fusion and attribution analysis.
All sensor observations, processing steps, and derived products are
recorded in a tamper-evident chain of custody with SHA-256 cryptographic
hashing. Any modification to the underlying data or processing chain
will produce a hash mismatch detectable via the included manifest.

This evidence package is suitable for submission in regulatory
enforcement proceedings, civil litigation, and carbon credit
invalidation actions.
================================================================================
"""


@dataclass
class ManifestEntry:
    """An entry in the evidence package manifest."""

    filename: str
    description: str
    sha256: str
    size_bytes: int
    content_type: str


@dataclass
class EvidencePackage:
    """A complete, sealed court-ready evidence package.

    Contains all artifacts needed for legal submission, with a
    cryptographic manifest ensuring tamper-evidence.
    """

    package_id: str
    case_id: str
    created_at: datetime
    report: AttributionReport

    # Package artifacts (filename → content bytes)
    artifacts: Dict[str, bytes] = field(default_factory=dict)

    # Cryptographic manifest
    manifest: List[ManifestEntry] = field(default_factory=list)

    # Package seal: SHA-256 of concatenated manifest hashes
    seal: Optional[str] = None

    def compute_seal(self) -> str:
        """Compute the package seal from all manifest entry hashes."""
        combined = "".join(sorted(e.sha256 for e in self.manifest))
        self.seal = hashlib.sha256(combined.encode()).hexdigest()
        return self.seal

    def verify_seal(self) -> bool:
        """Verify the package seal matches the current manifest."""
        if not self.seal:
            return False
        combined = "".join(sorted(e.sha256 for e in self.manifest))
        expected = hashlib.sha256(combined.encode()).hexdigest()
        return self.seal == expected

    def verify_artifacts(self) -> List[str]:
        """Verify all artifacts match their manifest hashes.

        Returns list of error messages (empty = all valid).
        """
        errors = []
        manifest_map = {e.filename: e for e in self.manifest}

        for filename, content in self.artifacts.items():
            if filename not in manifest_map:
                errors.append(f"{filename}: not in manifest")
                continue

            expected_hash = manifest_map[filename].sha256
            actual_hash = hashlib.sha256(content).hexdigest()
            if actual_hash != expected_hash:
                errors.append(
                    f"{filename}: hash mismatch "
                    f"(expected {expected_hash[:16]}..., got {actual_hash[:16]}...)"
                )

            expected_size = manifest_map[filename].size_bytes
            if len(content) != expected_size:
                errors.append(
                    f"{filename}: size mismatch "
                    f"(expected {expected_size}, got {len(content)})"
                )

        for entry in self.manifest:
            if entry.filename not in self.artifacts:
                errors.append(f"{entry.filename}: in manifest but missing from artifacts")

        return errors

    def to_summary(self) -> dict:
        """Summary for display/logging."""
        return {
            "package_id": self.package_id,
            "case_id": self.case_id,
            "created_at": self.created_at.isoformat(),
            "report_id": self.report.report_id,
            "artifact_count": len(self.artifacts),
            "manifest_entries": len(self.manifest),
            "seal": self.seal,
            "seal_valid": self.verify_seal(),
        }

    def write_to_directory(self, output_dir: str) -> str:
        """Write all artifacts to a directory on disk.

        Returns the path to the output directory.
        """
        out = Path(output_dir) / self.package_id
        out.mkdir(parents=True, exist_ok=True)

        for filename, content in self.artifacts.items():
            filepath = out / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_bytes(content)

        return str(out)


class EvidencePackager:
    """Assembles court-ready evidence packages.

    Takes attribution reports and bundles them with all supporting
    evidence into a sealed, tamper-evident package.
    """

    def __init__(self) -> None:
        self._env = Environment(loader=BaseLoader())
        self._template = self._env.from_string(REPORT_TEMPLATE)

    def package(
        self,
        report: AttributionReport,
        chain: ChainOfCustody,
        additional_artifacts: Optional[Dict[str, bytes]] = None,
    ) -> EvidencePackage:
        """Create a sealed evidence package from an attribution report.

        Args:
            report: The attribution report to package.
            chain: The chain of custody for this case.
            additional_artifacts: Optional extra files to include
                (e.g., visualization PNGs, raw data references).

        Returns:
            A sealed EvidencePackage ready for legal submission.
        """
        package = EvidencePackage(
            package_id=f"PKG-{uuid4().hex[:8].upper()}",
            case_id=report.case_id,
            created_at=datetime.now(timezone.utc),
            report=report,
        )

        # 1. Render human-readable report
        rendered = self._render_report(report)
        package.artifacts["attribution_report.txt"] = rendered.encode("utf-8")

        # 2. Machine-readable JSON evidence bundle
        evidence_json = json.dumps(report.to_dict(), indent=2, default=str)
        package.artifacts["evidence_bundle.json"] = evidence_json.encode("utf-8")

        # 3. Chain of custody record
        chain_json = chain.to_json()
        package.artifacts["chain_of_custody.json"] = chain_json.encode("utf-8")

        # 4. Additional artifacts
        if additional_artifacts:
            for name, content in additional_artifacts.items():
                package.artifacts[name] = content

        # 5. Build cryptographic manifest
        for filename, content in package.artifacts.items():
            entry = ManifestEntry(
                filename=filename,
                description=self._describe_artifact(filename),
                sha256=hashlib.sha256(content).hexdigest(),
                size_bytes=len(content),
                content_type=self._content_type(filename),
            )
            package.manifest.append(entry)

        # 6. Manifest as artifact
        manifest_json = json.dumps(
            [
                {
                    "filename": e.filename,
                    "description": e.description,
                    "sha256": e.sha256,
                    "size_bytes": e.size_bytes,
                    "content_type": e.content_type,
                }
                for e in package.manifest
            ],
            indent=2,
        )
        manifest_bytes = manifest_json.encode("utf-8")
        package.artifacts["manifest.json"] = manifest_bytes
        # Add manifest itself to the manifest list
        package.manifest.append(ManifestEntry(
            filename="manifest.json",
            description="Cryptographic manifest of all package artifacts",
            sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            size_bytes=len(manifest_bytes),
            content_type="application/json",
        ))

        # 7. Compute package seal
        package.compute_seal()

        # 8. Record packaging in chain of custody
        chain.add_link(
            action=CustodyAction.PACKAGE,
            actor="pmpe.evidence.packager",
            description=(
                f"Assembled court-ready evidence package {package.package_id} "
                f"containing {len(package.artifacts)} artifacts — "
                f"seal: {package.seal[:16]}..."
            ),
            input_hashes=[e.sha256 for e in package.manifest],
            output_hash=package.seal,
            metadata={
                "package_id": package.package_id,
                "artifact_count": len(package.artifacts),
                "seal": package.seal,
            },
        )

        return package

    def _render_report(self, report: AttributionReport) -> str:
        """Render attribution report to human-readable text."""
        return self._template.render(
            case_id=report.case_id,
            report=report,
        )

    @staticmethod
    def _describe_artifact(filename: str) -> str:
        descriptions = {
            "attribution_report.txt": "Human-readable attribution report",
            "evidence_bundle.json": "Machine-readable evidence data",
            "chain_of_custody.json": "Tamper-evident chain of custody record",
            "manifest.json": "Cryptographic manifest of all package artifacts",
        }
        return descriptions.get(filename, f"Supporting artifact: {filename}")

    @staticmethod
    def _content_type(filename: str) -> str:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        types = {
            "json": "application/json",
            "txt": "text/plain",
            "png": "image/png",
            "pdf": "application/pdf",
            "tif": "image/tiff",
            "tiff": "image/tiff",
            "nc": "application/x-netcdf",
        }
        return types.get(ext, "application/octet-stream")
