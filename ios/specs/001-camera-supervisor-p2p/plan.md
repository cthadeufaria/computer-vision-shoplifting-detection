# Implementation Plan: iOS Smart Camera to Supervisory Device P2P App

**Branch**: `001-camera-supervisor-p2p` | **Date**: 2026-04-09 | **Spec**: [/Users/bernese/git/computer-vision-shoplifting-detection/ios/specs/001-camera-supervisor-p2p/spec.md](/Users/bernese/git/computer-vision-shoplifting-detection/ios/specs/001-camera-supervisor-p2p/spec.md)
**Input**: Feature specification from `/Users/bernese/git/computer-vision-shoplifting-detection/ios/specs/001-camera-supervisor-p2p/spec.md`

## Summary

Extend the existing Swift/SwiftUI iOS app so one device can operate as a smart camera and up to four other feeds can be monitored by a supervisor device over authenticated local-LAN peer-to-peer connections. The implementation keeps MVVM boundaries intact, uses injected services for camera, pose estimation, inference, pairing, and persistence, and adopts the latest local STG-NF training run at `artifacts/stg_nf/multi_runs/Multi/Apr01_1416/Apr01_1419__checkpoint.pth.tar` as the source model for the bundled CoreML package.

## Technical Context

**Language/Version**: Swift 6.0 for the app, Python 3 for fixture/model conversion scripts  
**Primary Dependencies**: SwiftUI, Combine, AVFoundation, Vision, CoreML, Network.framework, CoreImage, XCTest, XCUITest, XcodeGen  
**Storage**: UserDefaults via `PersistenceService`; bundled `.mlpackage` and JSON fixtures in app/test resources; no database  
**Testing**: XCTest unit and integration tests, XCUITest UI tests, `xcodebuild test`, fixture comparisons against Python-generated JSON  
**Target Platform**: iOS 15.0+ on iPhone and iPad, Xcode 16, physical devices required for camera/local-network validation  
**Project Type**: Native iOS mobile app  
**Performance Goals**: 30 fps capture on iPhone 8+, <50 ms CoreML inference per 24-frame window, supervisor streaming capped at 10 fps, anomaly badge propagation within 500 ms  
**Constraints**: MVVM with dependency injection, TDD-first, Swift 6 strict concurrency, no UIKit/AVFoundation/Vision in ViewModels, no frame storage to disk, QR token valid only while pairing screen is visible, max four simultaneous supervisor feeds, current v1 transport remains local LAN only  
**Scale/Scope**: One camera-to-one supervisor session per camera device, one supervisor viewing up to four feeds, four onboarding screens, one detection workflow, one supervisor grid/full-screen workflow, one bundled STG-NF model

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **I. MVVM Architecture**: Pass with required follow-up. The feature plan keeps business logic in services and ViewModels, and explicitly adds protocol-backed `PairingService`, `StreamingService`, `TrackingService`, `PermissionService`, and `PersistenceService` injection to remove remaining hard-coded dependencies identified in `AGENTS.md`.
- **II. Test-First Development (TDD)**: Pass. The plan requires networking, ViewModel, model-conversion, and UI-flow tests before corresponding implementation changes.
- **III. iOS 15+ Compatibility & Performance**: Fails in current repo, remediation defined. [`project.yml`](/Users/bernese/git/computer-vision-shoplifting-detection/ios/project.yml) still targets iOS 17.0, while the constitution and feature spec require iOS 15.0. Implementation must first lower deployment target in project config and preserve iOS 15-compatible APIs before feature work is considered complete.
- **IV. Swift Best Practices & Code Quality**: Pass with follow-up. Plan uses strict concurrency-safe services/actors and protocol abstractions; SwiftLint integration remains an implementation task.
- **V. Numeric Fidelity**: Pass with explicit enforcement. The plan uses fixture-backed validation, exact `opp_order`, IoU tracker parameters `iou_threshold=0.3` and `max_missing=6`, and the latest local Multi run checkpoint/config as the authoritative model source.
- **VI. Privacy & Data Transmission**: Partial pass, implementation constraint noted. Constitution requires encrypted authenticated transport, while the current v1 spec says local-LAN only and existing app plan assumes raw TCP. Phase 1 contracts therefore define authenticated pairing plus a transport seam so v1 can ship on local LAN while leaving room for an encrypted `NWProtocolTLS` upgrade during implementation without changing higher-level services.

**Gate Decision**: Proceed to planning with explicit remediation tasks. No unresolved clarification blocks remain, but implementation must treat iOS 15 compatibility and encrypted/authenticated transport as mandatory gates.

## Project Structure

### Documentation (this feature)

```text
/Users/bernese/git/computer-vision-shoplifting-detection/ios/specs/001-camera-supervisor-p2p/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
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
│   ├── Networking/              # to add for pairing/streaming/session types
│   ├── Onboarding/
│   ├── Supervisor/              # to add for feed grid/detail flows
│   └── Resources/
├── ShopliftDetectTests/
│   ├── Camera/
│   ├── Detection/
│   ├── Fixtures/
│   ├── Mocks/
│   ├── Model/
│   ├── Networking/              # to add
│   ├── Onboarding/
│   └── Pose/
├── ShopliftDetectUITests/
├── scripts/
└── specs/
```

**Structure Decision**: Use the existing single iOS app project rooted at `/Users/bernese/git/computer-vision-shoplifting-detection/ios`, extending the current `ShopliftDetect` target with new `Networking/` and `Supervisor/` modules rather than creating a second app or backend service. This preserves the present XcodeGen/Xcode project, matches the current repo layout, and minimizes churn while keeping MVVM/service boundaries explicit.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Deferred repo-wide iOS target fix | Current repo is already bootstrapped at iOS 17.0 | Stopping planning entirely would not reduce implementation risk; the plan instead makes lowering to iOS 15 the first mandatory task |
| Transport seam for authenticated + future encrypted streaming | Constitution requires encrypted/authenticated transport while current v1 scope is local LAN | Hardcoding raw unauthenticated TCP in ViewModels would violate the constitution and block later transport upgrades |

## Post-Design Constitution Check

- **MVVM**: Pass. New responsibilities are partitioned into `OnboardingViewModel`, `HomeViewModel`, `DetectionViewModel`, `SupervisorViewModel`, `PairingService`, `StreamingService`, `TrackingService`, `PermissionService`, and `PersistenceService`.
- **TDD**: Pass. Every new behavior in the feature spec maps to unit, integration, or UI tests, including loopback pairing and stale-tile behavior.
- **iOS 15+**: Pass in design, pending implementation. The design excludes iOS 16+ only APIs and explicitly requires changing XcodeGen/Xcode project settings and CoreML conversion target to iOS 15.
- **Swift Quality**: Pass. Actors are reserved for hot-path shared state (`FrameBuffer`, networking/session coordination as needed), and protocols front all hardware/inference services.
- **Numeric Fidelity**: Pass. Latest local checkpoint/config and existing fixtures remain authoritative; additional fixtures are required only if preprocessing/model parameters change.
- **Privacy/Transmission**: Pass in design. Pairing is authenticated with single-use tokens and the streaming boundary is specified as transport-agnostic so the implementation can satisfy encryption requirements without redesigning app layers.
