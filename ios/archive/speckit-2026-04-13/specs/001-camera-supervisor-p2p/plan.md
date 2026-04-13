# Implementation Plan: iOS Smart Camera to Supervisory Device P2P App

**Branch**: `001-camera-supervisor-p2p` | **Date**: 2026-04-11 | **Spec**: [/Users/bernese/git/computer-vision-shoplifting-detection/ios/specs/001-camera-supervisor-p2p/spec.md](/Users/bernese/git/computer-vision-shoplifting-detection/ios/specs/001-camera-supervisor-p2p/spec.md)
**Input**: Feature specification from `/Users/bernese/git/computer-vision-shoplifting-detection/ios/specs/001-camera-supervisor-p2p/spec.md`

## Summary

Extend the existing Swift/SwiftUI iOS app so one device can operate as a smart camera and another as a supervisor over authenticated local-LAN peer-to-peer connections, while also formalizing `Pose Preview` as a supported diagnostic workflow for camera-role operators. The implementation keeps MVVM boundaries intact, uses injected services for camera, pose estimation, inference, pairing, persistence, and reporting, and adds a post-delivery hardening phase that produces a feature coverage matrix mapping user-visible capabilities to implementation files, automated tests, and manual device-validation evidence.

## Technical Context

**Language/Version**: Swift 6.0 for the app, Python 3 for fixture/model conversion and reporting scripts  
**Primary Dependencies**: SwiftUI, Combine, AVFoundation, Vision, CoreML, Network.framework with TLS parameters, CoreImage, XCTest, XCUITest, XcodeGen, SwiftLint  
**Storage**: UserDefaults via `PersistenceService`; bundled `.mlpackage` and JSON fixtures in app/test resources; markdown planning artifacts in `specs/`; no database  
**Testing**: XCTest unit and integration tests, XCUITest UI tests, `xcodebuild test`, fixture comparisons against Python-generated JSON, manual physical-device checklist execution  
**Target Platform**: iOS 15.0+ on iPhone and iPad, Xcode 16, physical devices required for camera/local-network validation  
**Project Type**: Native iOS mobile app  
**Performance Goals**: 30 fps capture on iPhone 8+, <50 ms CoreML inference per 24-frame window, supervisor streaming capped at 10 fps, anomaly badge propagation within 500 ms, Pose Preview open-dismiss-start-detection flow completed in under 30 seconds  
**Constraints**: MVVM with dependency injection, TDD-first, Swift 6 strict concurrency, no UIKit/AVFoundation/Vision in ViewModels, no frame storage to disk, QR token valid only while pairing screen is visible, max four simultaneous supervisor feeds, encrypted/authenticated local-LAN transport only in v1, feature coverage matrix must be generated from repo artifacts and manual validation evidence  
**Scale/Scope**: One camera-to-one supervisor session per camera device, one supervisor viewing up to four feeds, four onboarding screens, one detection workflow, one pose-preview workflow, one supervisor grid/full-screen workflow, one bundled STG-NF model, one feature-coverage reporting artifact

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **I. MVVM Architecture**: Pass with explicit follow-up. The design keeps rendering in SwiftUI views, state orchestration in `@MainActor` ViewModels, and camera/pose/network/reporting behavior in injected services or scripts.
- **II. Test-First Development (TDD)**: Pass with explicit follow-up. Each feature slice and the new hardening/reporting work must begin with failing unit/UI/reporting tests or validation checks before implementation.
- **III. iOS 15+ Compatibility & Performance**: Pass with follow-up. `project.yml` targets iOS 15.0, but completion still requires physical-device validation for detection throughput, Pose Preview stability, and sustained-session performance against constitution thresholds.
- **IV. Swift Best Practices & Code Quality**: Pass with explicit follow-up. Strict concurrency, protocol-backed hardware boundaries, and SwiftLint enforcement remain mandatory gates; new coverage-reporting logic belongs in scripts/docs, not view logic.
- **V. Numeric Fidelity**: Pass. Existing fixture-backed validation, exact `opp_order`, and exact IoU tracker settings remain authoritative for preprocessing and model behavior.
- **VI. Privacy & Data Transmission**: Pass. Transport remains encrypted/authenticated and local-LAN scoped for v1; no new reporting artifact may capture or persist video-frame data.

**Gate Decision**: Proceed. No unresolved clarifications block planning. Remaining completion gates are physical-device validation, quantified performance evidence, and maintaining artifact consistency between spec, plan, tasks, and coverage reporting.

## Project Structure

### Documentation (this feature)

```text
/Users/bernese/git/computer-vision-shoplifting-detection/ios/specs/001-camera-supervisor-p2p/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── feature-coverage-matrix.md      # to add during hardening/reporting
├── contracts/
│   ├── pairing-protocol.md
│   └── stream-protocol.md
└── tasks.md
```

### Source Code (repository root)

```text
/Users/bernese/git/computer-vision-shoplifting-detection/ios/
├── project.yml
├── ShopliftDetect.xcodeproj
├── ShopliftDetect/
│   ├── App/
│   ├── Core/
│   │   ├── Camera/
│   │   ├── Domain/
│   │   ├── Model/
│   │   ├── Pose/
│   │   ├── Protocols/
│   │   ├── Services/
│   │   └── UI/
│   ├── Detection/
│   ├── Home/
│   ├── Networking/
│   ├── Onboarding/
│   ├── Supervisor/
│   └── Resources/
├── ShopliftDetectTests/
│   ├── Camera/
│   ├── Detection/
│   ├── Fixtures/
│   ├── Mocks/
│   ├── Model/
│   ├── Networking/
│   ├── Onboarding/
│   ├── Pose/
│   └── Supervisor/
├── ShopliftDetectUITests/
├── scripts/
│   └── generate_feature_coverage_matrix.py   # to add during hardening/reporting
└── specs/
```

**Structure Decision**: Keep the existing single-target iOS app. Extend the current module layout with no new app targets, preserve SwiftUI + MVVM boundaries, and treat coverage reporting as a documentation/script concern under `specs/` and `scripts/`, not as runtime app code.

## Design Phases

### Phase 0: Research Baseline

- Reuse the existing decisions in [`research.md`](/Users/bernese/git/computer-vision-shoplifting-detection/ios/specs/001-camera-supervisor-p2p/research.md) as the source of truth for deployment target, model provenance, transport design, token lifecycle, feed cap, and threshold persistence.
- No new technical unknowns remain for Pose Preview scope; it uses the same camera and pose-estimation stack as camera detection, but without anomaly scoring UI.

### Phase 1: Core Feature Design

- Camera-role workflows: onboarding, QR pairing display, Pose Preview diagnostics, detection, threshold calibration, and outbound supervisor stream publishing.
- Supervisor workflows: onboarding, QR scanning, authenticated pairing, live grid monitoring, full-screen expansion, stale/disconnected overlays.
- Shared design contracts: pairing protocol, framed streaming protocol, persisted role/settings models, per-track detection results, and session state transitions.

### Phase 2: Hardening, Reporting, and Validation

- Extend the delivery plan with an explicit stabilization phase after the initial feature set is functional.
- Produce `feature-coverage-matrix.md` as a requirement-to-artifact report mapping:
  - functional requirements and user-visible flows
  - implementation files
  - automated unit/UI/integration tests
  - manual validation steps from `quickstart.md`
  - remaining gaps or unsupported evidence
- Add a helper script in `scripts/` to derive matrix inputs from repo structure and current artifact references.
- Treat the matrix as an execution/reporting deliverable, not as a substitute for line coverage tools. It is feature coverage, not compiler or XCTest percentage coverage.

## Data Model Alignment

- Existing entities in [`data-model.md`](/Users/bernese/git/computer-vision-shoplifting-detection/ios/specs/001-camera-supervisor-p2p/data-model.md) remain valid for detection, pairing, and supervisor workflows.
- `Pose Preview` introduces no new persistent entities; it reuses `PoseSkeleton`, `Keypoint`, camera session state, and the Smart Camera role context.
- The feature coverage matrix introduces one documentation artifact rather than a runtime model. Its logical rows are derived from `FR-*`, `SC-*`, user stories, implementation paths, test paths, and manual validation references.

## Contracts and Interfaces

- [`contracts/pairing-protocol.md`](/Users/bernese/git/computer-vision-shoplifting-detection/ios/specs/001-camera-supervisor-p2p/contracts/pairing-protocol.md) remains the source of truth for QR payload and authenticated handshake behavior.
- [`contracts/stream-protocol.md`](/Users/bernese/git/computer-vision-shoplifting-detection/ios/specs/001-camera-supervisor-p2p/contracts/stream-protocol.md) remains the source of truth for heartbeats, frames, detection payloads, and disconnect handling.
- No external contract is needed for Pose Preview; it is an in-app diagnostic interface.
- The new reporting deliverable requires a lightweight documentation contract:
  - each matrix row references a requirement or user-visible flow
  - each row lists code files, automated tests, and manual validation evidence
  - each row marks coverage status as covered, partially covered, or manual-only

## Deliverables

- Updated implementation plan aligned to `FR-020`, `FR-021`, and `SC-010`
- Existing feature artifacts preserved:
  - `research.md`
  - `data-model.md`
  - `contracts/pairing-protocol.md`
  - `contracts/stream-protocol.md`
  - `quickstart.md`
- New post-delivery artifact to add during hardening:
  - `feature-coverage-matrix.md`
- Supporting automation/script artifact to add during hardening:
  - `scripts/generate_feature_coverage_matrix.py`

## Validation Strategy

- Automated validation:
  - required suite inventory for `SC-008`:
    - `ShopliftDetectTests` via `xcodebuild test -only-testing:ShopliftDetectTests`
    - `ShopliftDetectUITests` via full-scheme `xcodebuild test`
    - fixture-backed numeric fidelity coverage in `STGNFModelIntegrationTests` and `PoseNormalizerTests`
    - networking/integration coverage for pairing, stream protocol, and encrypted transport tests
  - unit tests for ViewModels, scoring, normalization, pairing, streaming, tracking, and new pose-preview lifecycle coverage
  - UI tests for onboarding, detection toggle, Pose Preview, and supervisor workflows
- Manual validation:
  - physical-device checklist in `quickstart.md`
  - measured elapsed-time capture for `SC-004` QR pairing and `SC-010` Pose Preview open-dismiss-start-detection workflow
  - repeated Pose Preview present/dismiss cycles
  - sustained Pose Preview stability
  - detection startup after Pose Preview use
  - pairing, supervisor feed update, and stale/disconnect verification
- Reporting validation:
  - coverage matrix must reference all formal-scope stories, including Pose Preview
  - matrix must identify any requirement with only manual validation or missing automated coverage
  - matrix must record measured times for onboarding-to-detection, QR pairing, and Pose Preview recovery workflows when those success criteria apply

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Feature coverage matrix artifact | The project needs feature-level traceability across spec, code, tests, and manual validation before the next hardening cycle | Raw line coverage alone does not show whether user-visible behavior is actually covered |
| Post-delivery hardening phase | Initial feature delivery is complete enough to validate, but stability and device evidence remain incomplete | Folding all hardening into the original story phases obscures release readiness and artifact ownership |

## Post-Design Constitution Check

- **MVVM**: Pass. Pose Preview remains a Smart Camera diagnostic flow backed by an injected ViewModel and shared camera/pose services; no business logic is added to SwiftUI views.
- **TDD**: Pass with follow-up. The hardening phase explicitly requires new regression tests and report-generation validation before bug-fix completion.
- **iOS 15+**: Pass in design, pending verification. The plan keeps iOS 15 compatibility as a hard constraint and binds completion to physical-device evidence rather than simulator-only success.
- **Swift Quality**: Pass. Runtime code remains in Swift modules; reporting logic is isolated to scripts and markdown artifacts.
- **Numeric Fidelity**: Pass. Pose Preview reuses pose-detection output without changing normalization or model assumptions, so no new numeric contract is introduced.
- **Privacy/Transmission**: Pass. Coverage reporting and validation artifacts must not capture or persist frame payloads; they record only pass/fail evidence and artifact references.
