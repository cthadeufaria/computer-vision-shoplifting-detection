<!--
SYNC IMPACT REPORT
==================
Version change: 1.1.0 → 1.1.1

Amendment summary (v1.1.1 — alignment patch against ios-app-plan.md):

  Fixed conflicts:
  - Principle III + CoreML Conversion Gate: CoreML model MUST be compiled with
    minimum_deployment_target=iOS15, not iOS17 as the plan's conversion script
    currently specifies. Constitution now makes this explicit.
  - Implementation Order Gate: updated step count from "Steps 1–23" to "Steps 1–25"
    to reflect the networking additions (Steps 16–23) + final regression step.

  Fixed misalignments:
  - Numeric Fidelity (Principle V): added IoU tracker parameters (iou_threshold=0.3,
    max_missing=6) from Python's SimpleIoUTracker — these must match exactly,
    same class of fidelity concern as opp_order.

  Closed gaps:
  - Development Workflow: added Simulator vs On-Device test split rule.
  - Principle III: added anomaly threshold default (-1.2) and UserDefaults-backed
    calibration requirement, as flagged in plan's Known Issues section.

  Deferred TODOs (carried forward):
  - project.yml MUST be updated to `iphoneos_deployment_target: "15.0"` — currently
    states iOS 17+. Constitution governs; plan is the non-conformant document.
  - convert_stgnf_to_coreml.py MUST be updated: change
    `minimum_deployment_target=coremltools.target.iOS17` →
    `minimum_deployment_target=coremltools.target.iOS15`.

Previous version (1.1.0) sync impact:
Version change: 1.0.0 → 1.1.0
Modified principles: VI — "Local-Only Privacy" renamed to "Privacy & Data Transmission"

Amendment summary (v1.1.0):
  - Principle VI relaxed from "frames must never leave local network" to
    "frames must be encrypted and never stored on third-party servers".
  - Added explicit list of permitted future transports: Tailscale, signaling
    server + NAT traversal, self-hosted TURN relay.
  - Current v1 local-LAN behaviour preserved as the active implementation;
    future transports are permitted but not yet implemented.
  - Rationale updated to reflect encryption+non-persistence as the governing
    constraint rather than physical network boundary.

Previous version (1.0.0) sync impact:
Version change: (template) → 1.0.0
Modified principles: N/A — first authored version from blank template

Added sections:
  - I.   MVVM Architecture
  - II.  Test-First Development (TDD)
  - III. iOS 15+ Compatibility & Performance
  - IV.  Swift Best Practices & Code Quality
  - V.   Numeric Fidelity
  - VI.  Local-Only Privacy
  - Performance Standards
  - Development Workflow

Removed sections: N/A

Templates requiring updates:
  ✅ .specify/templates/constitution-template.md — source template (read-only reference, no update needed)
  ✅ .specify/templates/plan-template.md — Constitution Check section aligns with principles defined here
  ✅ .specify/templates/spec-template.md — requirements framing aligns with MVVM + TDD constraints
  ✅ .specify/templates/tasks-template.md — TDD task order (write test → fail → implement) aligns with Principle II

Files requiring manual follow-up:
  ⚠ ../ios-app-plan.md — project.yml target must change from iOS 17+ to iOS 15+
  ⚠ scripts/convert_stgnf_to_coreml.py — minimum_deployment_target must change from iOS17 to iOS15
-->

# ShopliftDetect Constitution

## Core Principles

### I. MVVM Architecture (NON-NEGOTIABLE)

Every screen MUST follow the Model-View-ViewModel pattern without exception:

- **Views** MUST contain zero business logic. They bind to a ViewModel and render state.
- **ViewModels** MUST be independently instantiable and testable without a running UI.
  They expose `@Published` state and receive user intents as method calls.
- **Models** MUST be pure value types (`struct` or `enum`) that are `Sendable`.
  No `UIKit`/`SwiftUI` imports in model files.
- **Services / Core components** (`CameraSession`, `PoseEstimator`, `STGNFModel`, etc.)
  are injected into ViewModels via protocol abstractions to enable unit testing.
- Protocol abstractions MUST exist for every hardware-touching or inference component
  so tests can inject fakes without hitting real devices.

Rationale: MVVM separates rendering concerns from domain logic, enabling full TDD
coverage of all pipeline logic and making the architecture portable to a future
Rust compute layer without touching the UI.

### II. Test-First Development (NON-NEGOTIABLE)

TDD is mandatory for every piece of logic in this project. The cycle is:

1. Write the test. It MUST NOT compile or MUST fail — verify this before continuing.
2. Write the minimum implementation to make the test pass.
3. Refactor while keeping tests green.

No implementation file MUST be written before at least one failing test exists for it.
This applies to all layers: domain types, converters, normalizers, buffers, scorers,
ViewModels, networking, and UI flows.

Specific rules:
- Unit tests (XCTest) MUST be written for: `KeypointConverter`, `PoseNormalizer`,
  `FrameBuffer`, `AnomalyScorer`, all ViewModels, `PairingService`, `StreamProtocol`.
- Integration tests MUST be written for: `STGNFModelIntegrationTests` (skip-guarded
  until `.mlpackage` is present), two-device loopback networking tests.
- UI tests (XCUITest) MUST be written for: onboarding flow, role selection, detection
  toggle, role persistence.
- Fixture-backed tests MUST compare Swift output against Python-generated JSON fixtures
  at `ShopliftDetectTests/Fixtures/` with the tolerances specified in Principle V.

Rationale: The app ports numerically sensitive Python algorithms (keypoint reorder,
mean/std normalization, NLL scoring). Without TDD, subtle off-by-one or axis errors
produce silently wrong anomaly scores that are impossible to catch via manual testing.

### III. iOS 15+ Compatibility & Performance

The minimum deployment target is **iOS 15.0**. This is a hard constraint that governs
all components: the Xcode project, the SwiftUI API surface, and the CoreML model.

- `project.yml` MUST declare `iphoneos_deployment_target: "15.0"`.
- `scripts/convert_stgnf_to_coreml.py` MUST use
  `minimum_deployment_target=coremltools.target.iOS15` — NOT iOS17.
- No API introduced after iOS 15.0 MUST be used unconditionally. Use `@available`
  guards with a working fallback for any iOS 16+ or iOS 17+ API.
- SwiftUI features unavailable on iOS 15 MUST NOT be used (e.g., `NavigationStack`
  requires iOS 16 — use `NavigationView` instead).
- **Camera pipeline** MUST sustain 30 fps capture on an iPhone 8 (A11, 2017) or newer.
- **CoreML inference** MUST complete in under 50 ms per 24-frame window.
- **Frame streaming** to the supervisor device MUST NOT exceed 10 fps to remain within
  LAN bandwidth without congestion.
- Memory allocations on hot paths (camera callback, pose estimation, normalization)
  MUST be minimized. Prefer pre-allocated buffers over per-frame heap allocation.
- `FrameBuffer` MUST be a Swift `actor` to avoid lock contention on the main thread.
- `MLMultiArray` backing buffers SHOULD be reused across inference calls where possible.
- The anomaly score threshold MUST default to `-1.2` and MUST be persisted in
  `UserDefaults` so it can be calibrated per deployment without a code change.
  A Settings sheet MUST expose threshold adjustment before the app is considered
  production-ready (the ShanghaiTech model may score normal retail footage as
  anomalous at the default threshold).

Rationale: The app targets security/retail deployments where hardware may be several
years old. A performance regression that causes dropped frames or thermal throttling
on an older device invalidates the real-time detection use case entirely.

### IV. Swift Best Practices & Code Quality

- **Swift 6 strict concurrency** is enabled. All `@Sendable` and actor-isolation
  requirements MUST be satisfied — no `nonisolated(unsafe)` escapes without a
  documented justification.
- **SwiftLint** MUST be integrated and run as a build phase. Zero SwiftLint warnings
  are permitted to merge into `main`. The `.swiftlint.yml` ruleset MUST include at
  minimum: `force_unwrapping`, `implicitly_unwrapped_optional`,
  `force_try`, `line_length`, `function_body_length`, `cyclomatic_complexity`.
- `guard let` / `if let` MUST be preferred over force-unwrap (`!`) at all sites
  except locations where a `nil` result is provably impossible and is documented.
- Value types (`struct`) MUST be preferred over reference types (`class`) for domain
  models and data containers. `class` is only appropriate for `ObservableObject`
  ViewModels and reference-semantic services (`STGNFModel`, `CameraSession`).
- Protocol-oriented design MUST be used to abstract hardware dependencies.
  Every hardware or CoreML component MUST conform to a testable protocol.
- `async`/`await` and `Combine` publishers MUST not be mixed arbitrarily.
  The camera pipeline uses `Combine` (`pixelBufferPublisher`); new async/await
  code should use Swift Concurrency (`async`/`await`, `AsyncStream`).

Rationale: Swift 6 strict concurrency prevents data races in a multi-actor pipeline
that runs camera capture, inference, UI updates, and network streaming concurrently.
SwiftLint enforcement ensures the codebase remains readable across team members.

### V. Numeric Fidelity

Swift preprocessing logic MUST produce output that matches the Python reference
implementation within defined tolerances:

| Component | Tolerance | Reference |
|-----------|-----------|-----------|
| `KeypointConverter` (COCO17→COCO18 reorder) | exact integer index match | `dataset.py:keypoints17_to_coco18` |
| `PoseNormalizer` (mean/std normalization) | ≤ 1e-5 per element | `data_utils.py:normalize_pose` |
| `STGNFModel` NLL output vs Python | ≤ 1e-3 absolute | Python inference on same seed |
| `DetectionViewModel` IoU tracker | exact parameter match | `pipeline/video_inference_pipeline.py:SimpleIoUTracker` |

The `opp_order` array `[0,17,6,8,10,5,7,9,12,14,16,11,13,15,2,1,4,3]` MUST be
hard-coded exactly. Any deviation silently breaks the model input contract.

`PoseNormalizer` MUST implement these steps in order:
1. Divide all x by frame width, all y by frame height; confidence unchanged.
2. Compute spatial mean over all 24×18 joint (x, y) pairs.
3. Subtract mean from all (x, y).
4. Compute std on the y-column only (all 24×18 y values after subtraction).
5. Divide both x and y by that scalar std.
6. Transpose to shape `[1, 2, 24, 18]`, drop confidence channel → `MLMultiArray`.

The multi-person IoU tracker MUST use `iou_threshold=0.3` and `max_missing=6`,
matching `pipeline/video_inference_pipeline.py:SimpleIoUTracker` exactly.
Any change to these values constitutes a numeric fidelity break and requires
a new fixture set and explicit constitution amendment.

Fixture tests in `ShopliftDetectTests/Fixtures/` are the authoritative numerical
validation gate and MUST pass before any related implementation is merged.

Rationale: The CoreML model was trained in Python. Any mismatch in preprocessing
produces wrong anomaly scores. Tolerance thresholds are derived from float32
precision and the ONNX intermediate conversion rounding.

### VI. Privacy & Data Transmission

Video frames and user data MUST be transmitted only over encrypted, authenticated
connections. The current implementation uses local LAN only; future transports
(e.g. Tailscale, signaling-server NAT traversal) are permitted provided they
satisfy the constraints below.

**Non-negotiable across all transports:**
- Video frames MUST be transmitted only over an encrypted channel (TLS, WireGuard,
  or equivalent). Unencrypted internet routing of frames is prohibited.
- No third-party analytics, crash-reporting, or telemetry SDKs MUST be added.
- Camera frames MUST NOT be written to disk (no photo library saves, no temp files)
  except as an explicit, user-initiated action in a future feature.
- Video frames MUST NOT be stored on any third-party server. A relay server
  (e.g. TURN) may forward frames transiently but MUST NOT persist them.
- Signaling or coordination servers (used only for device pairing / NAT traversal)
  MUST NOT receive or forward video frame data — only connection metadata
  (IP addresses, ports, pairing codes).

**Current implementation (v1 — local LAN):**
- Video frames are transmitted over local LAN TCP connections (port 7890)
  established via QR-code pairing. No internet routing.
- The `NSLocalNetworkUsageDescription` and `NSCameraUsageDescription` `Info.plist`
  entries MUST accurately describe the scope of data use for the active transport.

**Future transports (permitted, not yet implemented):**
- Tailscale (WireGuard-based mesh VPN) — frames remain E2E encrypted; permitted.
- Signaling server + UDP hole punching — frames go peer-to-peer after handshake;
  permitted provided the signaling server receives no frame data.
- TURN relay fallback — permitted only if relay is self-hosted or contractually
  prohibited from storing data.

Rationale: The app is deployed in retail security contexts with strict data
protection obligations. The constraint is on encryption and non-persistence of
frame data, not on the physical network path — enabling future remote monitoring
use cases without compromising privacy guarantees.

## Performance Standards

| Metric | Requirement | Measurement Method |
|--------|-------------|-------------------|
| Camera capture frame rate | ≥ 30 fps on iPhone 8 (iOS 15) | Instruments / `CADisplayLink` counter |
| CoreML inference latency | ≤ 50 ms per 24-frame window | `testSingleWindowInferenceUnder50ms` |
| Warmup duration | ≤ 0.8 s at 30 fps (24 frames) | Frame counter in `FrameBuffer` |
| LAN streaming frame rate | ≤ 10 fps (supervisor path) | `StreamProtocol` send interval |
| App launch to camera active | ≤ 2 s on iPhone 8 | Manual device measurement |
| Memory footprint (steady state) | ≤ 150 MB | Instruments Allocations |

If a performance regression breaks any threshold above, it MUST be fixed before
merging to `main`, not deferred.

## Development Workflow

### TDD Cycle (enforced for every task)

```
1. Write failing test  →  2. Confirm it fails  →  3. Implement  →  4. Confirm passes  →  5. Refactor
```

Tasks template MUST order test writing before implementation within every phase.
No implementation task MAY be marked complete until its corresponding test passes.

### Linting Gate

SwiftLint MUST run as an Xcode build phase. The CI check (or pre-commit hook)
MUST block merges if any SwiftLint violation is present.
`.swiftlint.yml` lives at `ios/.swiftlint.yml` and is committed to the repository.

### Simulator vs On-Device Test Constraints

Not all tests can run on the simulator:

| Test category | Runs on simulator | Runs on device |
|---------------|:-----------------:|:--------------:|
| Unit tests (XCTest: converters, normalizers, buffers, scorers, ViewModels) | ✅ | ✅ |
| Integration tests (STGNFModelIntegrationTests — requires `.mlpackage`) | ✅ | ✅ |
| Networking loopback tests (PairingServiceTests, StreamProtocolTests) | ✅ | ✅ |
| UI tests (OnboardingUITests, RoleSelectionUITests, DetectionToggleUITests) | ✅ | ✅ |
| Camera + live pose tests (CameraSession, real VNDetectHumanBodyPoseRequest) | ❌ | ✅ |
| Two-device smoke test (LAN streaming, camera → supervisor) | ❌ | ✅ |

Camera and live pose tests MUST run on a physical device. The two-device smoke test
(iPhone camera role + iPad supervisor role on the same Wi-Fi) MUST be executed
before any networking milestone is declared complete.

### Implementation Order Gate

The implementation order defined in `../ios-app-plan.md` (Steps 1–25) is the
authoritative delivery sequence. No step MUST be started until its gate condition
(compilation, test pass, manual device test) is satisfied.

### CoreML Conversion Gate

`scripts/convert_stgnf_to_coreml.py` MUST produce an `.mlpackage` compiled with
`minimum_deployment_target=coremltools.target.iOS15` whose NLL output differs from
Python by ≤ 0.01 on a random input before any Swift inference code is written.
The conversion script output is the single source of truth for the model.

### Multi-Device Tests

Networking tests (`PairingServiceTests`, `StreamProtocolTests`) MUST run on loopback
first. The two-device smoke test (iPhone camera + iPad supervisor on LAN) MUST be
executed before any networking-related milestone is declared complete.

## Governance

- This constitution supersedes any conflicting guidance in README files, PR comments,
  or verbal agreements. When in doubt, the constitution is the authority.
- **Amendments** require: (1) a documented rationale, (2) version bump per semver
  rules (MAJOR for principle removal/redefinition, MINOR for additions, PATCH for
  clarifications), (3) update to `LAST_AMENDED_DATE`.
- **Compliance review**: every PR description MUST include a one-line "Constitution
  Check" confirming no principles are violated. If a violation is necessary, it MUST
  be justified in the Complexity Tracking table of the relevant `plan.md`.
- **Versioning policy**: semantic versioning applies.
  - MAJOR: removes or redefines a NON-NEGOTIABLE principle.
  - MINOR: adds a new principle, section, or materially expands guidance.
  - PATCH: clarifications, wording, tolerance adjustments, typo fixes.
- **Runtime guidance**: for agent-specific and workflow guidance see
  `.specify/integrations/claude.manifest.json` and the `.specify/templates/` directory.

**Version**: 1.1.1 | **Ratified**: 2026-04-08 | **Last Amended**: 2026-04-08
