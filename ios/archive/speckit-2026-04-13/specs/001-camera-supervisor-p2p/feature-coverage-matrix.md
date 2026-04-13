# Feature Coverage Matrix

Last updated: 2026-04-11

## Status Legend

- `covered`: implementation and automated validation are present, with no known open evidence gap
- `partial`: implementation exists, but either simulator execution evidence or physical-device evidence is still incomplete
- `manual-only`: no automated coverage exists yet

## Current Validation Evidence

### Automated Suites

| Date | Scope | Command / Inventory | Result | Notes |
|---|---|---|---|---|
| 2026-04-11 | Targeted unit regressions | `xcodebuild test -project ShopliftDetect.xcodeproj -scheme ShopliftDetect -destination 'platform=iOS Simulator,name=iPhone 17' -only-testing:ShopliftDetectTests/PosePreviewViewModelTests -only-testing:ShopliftDetectTests/DetectionViewModelTests -only-testing:ShopliftDetectTests/Networking/PairingServiceTests -only-testing:ShopliftDetectTests/Supervisor/SupervisorViewModelTests` | blocked | Simulator bootstrap exited early with `code 65` before test assertions completed. |
| 2026-04-11 | Targeted UI regressions | `xcodebuild test -project ShopliftDetect.xcodeproj -scheme ShopliftDetect -destination 'platform=iOS Simulator,name=iPhone 17' -only-testing:ShopliftDetectUITests/PosePreviewUITests -only-testing:ShopliftDetectUITests/OnboardingUITests` | unstable | Test runner restarted multiple times after unexpected app exits. Individual onboarding regressions executed and passed during reruns, but the selected-suite command did not finish with a clean final summary. |
| 2026-04-11 | Required suite inventory | `ShopliftDetectTests`, scheme-wide `ShopliftDetect` including `ShopliftDetectUITests`, plus high-signal integration coverage from `STGNFModelIntegrationTests`, `PoseNormalizerTests`, `PairingServiceTests`, `StreamProtocolTests`, and `EncryptedTransportTests` | open | `T059` remains open until the required simulator suite inventory completes cleanly. |

### Manual Device Evidence

| Date | Evidence | Result | Notes |
|---|---|---|---|
| 2026-04-11 | Baseline two-device validation (`T048`) | open | Physical-device checklist not run in this environment. |
| 2026-04-11 | Updated physical-device checklist (`T060`) | open | Pose Preview timing, pairing timing, and disconnect recovery timing still need device capture. |
| 2026-04-11 | On-device performance evidence (`T061`) | open | FPS, inference latency, thermal behavior, and memory footprint remain uncaptured. |

## Matrix

| Area | Requirements | Implementation | Automated Tests | Manual Evidence | Status | Notes |
|---|---|---|---|---|---|---|
| Onboarding and role persistence | FR-001<br>FR-002<br>SC-001 | `ShopliftDetect/Onboarding/OnboardingViewModel.swift`<br>`ShopliftDetect/Onboarding/OnboardingView.swift`<br>`ShopliftDetect/Home/HomeViewModel.swift` | `ShopliftDetectTests/Onboarding/OnboardingViewModelTests.swift`<br>`ShopliftDetectTests/Home/HomeViewModelTests.swift`<br>`ShopliftDetectUITests/OnboardingUITests.swift` | `quickstart.md` section 6 onboarding-to-detection timing capture | partial | UI coverage exists; physical-device timing evidence remains open. |
| Camera detection pipeline and threshold settings | FR-003<br>FR-004<br>FR-014<br>SC-002<br>SC-003 | `ShopliftDetect/Detection/DetectionViewModel.swift`<br>`ShopliftDetect/Core/Pose/PoseEstimator.swift`<br>`ShopliftDetect/Detection/TrackingService.swift`<br>`ShopliftDetect/Model/AnomalyScorer.swift` | `ShopliftDetectTests/Detection/DetectionViewModelTests.swift`<br>`ShopliftDetectTests/Detection/TrackingServiceTests.swift`<br>`ShopliftDetectTests/Model/AnomalyScorerTests.swift`<br>`ShopliftDetectUITests/DetectionToggleUITests.swift` | `quickstart.md` section 6 detection startup and supervisor overlay validation<br>`quickstart.md` section 7 fps / latency notes | partial | Session-invalidation guards were added to prevent stale in-flight frame updates after stop/restart. |
| Pose conversion and numeric fidelity | SC-009 | `ShopliftDetect/Core/Pose/PoseNormalizer.swift`<br>`ShopliftDetect/Core/Pose/KeypointConverter.swift`<br>`ShopliftDetect/Core/Model/STGNFModelWrapper.swift` | `ShopliftDetectTests/Pose/PoseNormalizerTests.swift`<br>`ShopliftDetectTests/Pose/KeypointConverterTests.swift`<br>`ShopliftDetectTests/Pose/KeypointConverterIntegrationTests.swift`<br>`ShopliftDetectTests/Model/STGNFModelIntegrationTests.swift` | - | covered | This area is fully automation-backed in the repo. |
| Camera pairing and token lifecycle | FR-005<br>FR-006<br>FR-007<br>FR-015<br>FR-017<br>SC-004 | `ShopliftDetect/Networking/PairingService.swift`<br>`ShopliftDetect/Networking/QRCodeDisplayView.swift`<br>`ShopliftDetect/Networking/QRScannerView.swift` | `ShopliftDetectTests/Networking/PairingServiceTests.swift`<br>`ShopliftDetectUITests/OnboardingUITests.swift` | `quickstart.md` section 6 QR scan start to connected state timing capture | partial | Recovery regressions for expired-token and invalid-token follow-up pairing were added. |
| Encrypted local transport and stream protocol | FR-008<br>FR-013<br>FR-018<br>SC-005<br>SC-006 | `ShopliftDetect/Networking/StreamingService.swift`<br>`ShopliftDetect/Networking/PairingService.swift`<br>`ShopliftDetect/Networking/SecureTransport.swift`<br>`ShopliftDetect/Networking/StreamProtocol.swift` | `ShopliftDetectTests/Networking/StreamProtocolTests.swift`<br>`ShopliftDetectTests/Networking/EncryptedTransportTests.swift` | `quickstart.md` section 6 supervisor receives frames and overlays | partial | Protocol and encryption checks exist; end-to-end multi-feed and latency evidence remains open. |
| Supervisor grid, tile expansion, and stale recovery | FR-009<br>FR-010<br>FR-011<br>FR-016 | `ShopliftDetect/Supervisor/SupervisorViewModel.swift`<br>`ShopliftDetect/Supervisor/SupervisorView.swift`<br>`ShopliftDetect/Supervisor/CameraFeedDetailView.swift` | `ShopliftDetectTests/Supervisor/SupervisorViewModelTests.swift`<br>`ShopliftDetectUITests/SupervisorMonitoringUITests.swift` | `quickstart.md` section 6 stale/disconnect overlay verification | partial | Stale-to-connected tile refresh coverage was added; disconnect timing still needs physical-device evidence. |
| Pose Preview diagnostic flow | FR-020<br>FR-021<br>SC-010 | `ShopliftDetect/Detection/PosePreviewViewModel.swift`<br>`ShopliftDetect/Detection/PosePreviewView.swift`<br>`ShopliftDetect/Home/HomeView.swift` | `ShopliftDetectTests/Detection/PosePreviewViewModelTests.swift`<br>`ShopliftDetectTests/Detection/DetectionViewModelTests.swift`<br>`ShopliftDetectUITests/PosePreviewUITests.swift` | `quickstart.md` section 6 repeated Pose Preview present/dismiss loop<br>`quickstart.md` section 6 Pose Preview 60-second stability check | partial | New lifecycle regressions were added, and stale in-flight preview updates are now dropped after stop/restart. |
| Lint and release gates | FR-019<br>SC-008 | `.swiftlint.yml`<br>`project.yml`<br>`ShopliftDetect.xcodeproj/project.pbxproj` | `SwiftLint` build phase<br>`ShopliftDetectTests`<br>`ShopliftDetectUITests` | `quickstart.md` section 4 required suite inventory | partial | Lint passes locally, but `T059` remains open until the required simulator suite inventory finishes cleanly. |

## Open Gaps

- `T048` is blocked on physical-device access for baseline two-device validation.
- `T059` is blocked on simulator stability because the latest targeted runs restarted after unexpected app exits and one unit-targeted command ended with an early bootstrap failure.
- `T060` is blocked on measured device timings for onboarding-to-detection, QR pairing, and Pose Preview recovery.
- `T061` is blocked on real-device performance capture for fps, latency, thermal behavior, and memory footprint.
