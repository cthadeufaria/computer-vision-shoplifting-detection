#!/usr/bin/env python3
"""Generate feature coverage matrix inputs for feature 001-camera-supervisor-p2p."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CoverageRow:
    area: str
    requirements: list[str]
    implementation: list[str]
    automated_tests: list[str]
    manual_evidence: list[str]
    status: str
    notes: str


ROWS = [
    CoverageRow(
        area="Onboarding and role persistence",
        requirements=["FR-001", "FR-002", "SC-001"],
        implementation=[
            "ShopliftDetect/Onboarding/OnboardingViewModel.swift",
            "ShopliftDetect/Onboarding/OnboardingView.swift",
            "ShopliftDetect/Home/HomeViewModel.swift",
        ],
        automated_tests=[
            "ShopliftDetectTests/Onboarding/OnboardingViewModelTests.swift",
            "ShopliftDetectTests/Home/HomeViewModelTests.swift",
            "ShopliftDetectUITests/OnboardingUITests.swift",
        ],
        manual_evidence=[
            "quickstart.md section 6 onboarding-to-detection timing capture",
        ],
        status="partial",
        notes="UI coverage exists; physical-device timing evidence remains open.",
    ),
    CoverageRow(
        area="Camera detection pipeline and threshold settings",
        requirements=["FR-003", "FR-004", "FR-014", "SC-002", "SC-003"],
        implementation=[
            "ShopliftDetect/Detection/DetectionViewModel.swift",
            "ShopliftDetect/Core/Pose/PoseEstimator.swift",
            "ShopliftDetect/Detection/TrackingService.swift",
            "ShopliftDetect/Model/AnomalyScorer.swift",
        ],
        automated_tests=[
            "ShopliftDetectTests/Detection/DetectionViewModelTests.swift",
            "ShopliftDetectTests/Detection/TrackingServiceTests.swift",
            "ShopliftDetectTests/Model/AnomalyScorerTests.swift",
            "ShopliftDetectUITests/DetectionToggleUITests.swift",
        ],
        manual_evidence=[
            "quickstart.md section 6 detection startup and supervisor overlay validation",
            "quickstart.md section 7 fps / latency notes",
        ],
        status="partial",
        notes="Regression coverage expanded; on-device performance evidence remains open.",
    ),
    CoverageRow(
        area="Pose conversion and numeric fidelity",
        requirements=["SC-009"],
        implementation=[
            "ShopliftDetect/Core/Pose/PoseNormalizer.swift",
            "ShopliftDetect/Core/Pose/KeypointConverter.swift",
            "ShopliftDetect/Core/Model/STGNFModelWrapper.swift",
        ],
        automated_tests=[
            "ShopliftDetectTests/Pose/PoseNormalizerTests.swift",
            "ShopliftDetectTests/Pose/KeypointConverterTests.swift",
            "ShopliftDetectTests/Pose/KeypointConverterIntegrationTests.swift",
            "ShopliftDetectTests/Model/STGNFModelIntegrationTests.swift",
        ],
        manual_evidence=[],
        status="covered",
        notes="This area is fully automation-backed.",
    ),
    CoverageRow(
        area="Camera pairing and token lifecycle",
        requirements=["FR-005", "FR-006", "FR-007", "FR-015", "FR-017", "SC-004"],
        implementation=[
            "ShopliftDetect/Networking/PairingService.swift",
            "ShopliftDetect/Networking/QRCodeDisplayView.swift",
            "ShopliftDetect/Networking/QRScannerView.swift",
        ],
        automated_tests=[
            "ShopliftDetectTests/Networking/PairingServiceTests.swift",
            "ShopliftDetectUITests/OnboardingUITests.swift",
        ],
        manual_evidence=[
            "quickstart.md section 6 QR scan start to connected state timing capture",
        ],
        status="partial",
        notes="Recovery tests were added; timed two-device validation remains open.",
    ),
    CoverageRow(
        area="Encrypted local transport and stream protocol",
        requirements=["FR-008", "FR-013", "FR-018", "SC-005", "SC-006"],
        implementation=[
            "ShopliftDetect/Networking/StreamingService.swift",
            "ShopliftDetect/Networking/PairingService.swift",
            "ShopliftDetect/Networking/SecureTransport.swift",
            "ShopliftDetect/Networking/StreamProtocol.swift",
        ],
        automated_tests=[
            "ShopliftDetectTests/Networking/StreamProtocolTests.swift",
            "ShopliftDetectTests/Networking/EncryptedTransportTests.swift",
        ],
        manual_evidence=[
            "quickstart.md section 6 supervisor receives frames and overlays",
        ],
        status="partial",
        notes="Protocol and encryption are covered in tests; end-to-end frame-rate evidence remains open.",
    ),
    CoverageRow(
        area="Supervisor grid, tile expansion, and stale recovery",
        requirements=["FR-009", "FR-010", "FR-011", "FR-016"],
        implementation=[
            "ShopliftDetect/Supervisor/SupervisorViewModel.swift",
            "ShopliftDetect/Supervisor/SupervisorView.swift",
            "ShopliftDetect/Supervisor/CameraFeedDetailView.swift",
        ],
        automated_tests=[
            "ShopliftDetectTests/Supervisor/SupervisorViewModelTests.swift",
            "ShopliftDetectUITests/SupervisorMonitoringUITests.swift",
        ],
        manual_evidence=[
            "quickstart.md section 6 stale/disconnect overlay verification",
        ],
        status="partial",
        notes="Recovery regression test added; device-backed disconnect evidence remains open.",
    ),
    CoverageRow(
        area="Pose Preview diagnostic flow",
        requirements=["FR-020", "FR-021", "SC-010"],
        implementation=[
            "ShopliftDetect/Detection/PosePreviewViewModel.swift",
            "ShopliftDetect/Detection/PosePreviewView.swift",
            "ShopliftDetect/Home/HomeView.swift",
        ],
        automated_tests=[
            "ShopliftDetectTests/Detection/PosePreviewViewModelTests.swift",
            "ShopliftDetectTests/Detection/DetectionViewModelTests.swift",
            "ShopliftDetectUITests/PosePreviewUITests.swift",
        ],
        manual_evidence=[
            "quickstart.md section 6 repeated Pose Preview present/dismiss loop",
            "quickstart.md section 6 Pose Preview 60-second stability check",
        ],
        status="partial",
        notes="Lifecycle regressions and stale in-flight updates were fixed; physical-device stability evidence remains open.",
    ),
    CoverageRow(
        area="Lint and release gates",
        requirements=["FR-019", "SC-008"],
        implementation=[
            ".swiftlint.yml",
            "project.yml",
            "ShopliftDetect.xcodeproj/project.pbxproj",
        ],
        automated_tests=[
            "SwiftLint build phase",
            "ShopliftDetectTests",
            "ShopliftDetectUITests",
        ],
        manual_evidence=[
            "quickstart.md section 4 required suite inventory",
        ],
        status="partial",
        notes="Lint passes locally; required suite inventory exists, but the latest simulator runs were unstable.",
    ),
]


def render_markdown(rows: list[CoverageRow]) -> str:
    lines = [
        "| Area | Requirements | Implementation | Automated Tests | Manual Evidence | Status | Notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {area} | {requirements} | {implementation} | {automated_tests} | {manual_evidence} | {status} | {notes} |".format(
                area=row.area,
                requirements="<br>".join(row.requirements),
                implementation="<br>".join(row.implementation),
                automated_tests="<br>".join(row.automated_tests) or "-",
                manual_evidence="<br>".join(row.manual_evidence) or "-",
                status=row.status,
                notes=row.notes,
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    args = parser.parse_args()

    if args.format == "json":
        print(json.dumps([asdict(row) for row in ROWS], indent=2))
        return

    print(render_markdown(ROWS))


if __name__ == "__main__":
    main()
