# Research: iOS Smart Camera to Supervisory Device P2P App

## Decision 1: Build on the existing native Swift/XcodeGen app instead of creating a new app target

- **Decision**: Extend the current `/Users/bernese/git/computer-vision-shoplifting-detection/ios` app using Swift 6, SwiftUI, XCTest, and XcodeGen.
- **Rationale**: The repository already contains a generated `ShopliftDetect.xcodeproj`, a populated `ShopliftDetect/` source tree, tests, fixtures, and a bundled `STGNFModel.mlpackage`. Planning against that structure reduces bootstrap churn and keeps work focused on missing networking, supervisor UI, and MVVM cleanup.
- **Alternatives considered**:
  - Create a second greenfield iOS app subtree: rejected because it duplicates existing setup and tests.
  - Use UIKit instead of SwiftUI: rejected because the current app already uses SwiftUI and the constitution standardizes MVVM around SwiftUI views.

## Decision 2: Treat iOS 15.0 as the required deployment target despite the current iOS 17 project settings

- **Decision**: Plan all implementation and API choices for iOS 15.0 compatibility, and make the `project.yml` target reduction from 17.0 to 15.0 an early gating task.
- **Rationale**: The constitution explicitly defines iOS 15.0 as non-negotiable, and the feature spec repeats that requirement. Allowing the plan to follow the current iOS 17 settings would produce invalid tasks and acceptance criteria.
- **Alternatives considered**:
  - Keep iOS 17 to match the current XcodeGen file: rejected because it violates the constitution and feature spec.
  - Add conditional support for two separate deployment targets: rejected as unnecessary complexity for a single-app PoC.

## Decision 3: Use the latest local Multi training run as the model source of truth

- **Decision**: Base model conversion and validation on `/Users/bernese/git/computer-vision-shoplifting-detection/artifacts/stg_nf/multi_runs/Multi/Apr01_1416/Apr01_1419__checkpoint.pth.tar` with configuration from `/Users/bernese/git/computer-vision-shoplifting-detection/artifacts/stg_nf/multi_runs/Multi/Apr01_1416/args.json`.
- **Rationale**: The user explicitly requested this run. `args.json` confirms the relevant inference parameters for the model family: dataset `Multi`, `seg_len=24`, `K=8`, `L=1`, `R=3.0`, `adj_strategy="uniform"`, and no confidence channel in model input.
- **Alternatives considered**:
  - Keep using the previously bundled model without tying it to a tracked run: rejected because it weakens reproducibility.
  - Switch to a different checkpoint family: rejected because the user specified the latest local Multi run.

## Decision 4: Keep the detection pipeline local to the camera device and stream annotated results outward

- **Decision**: Run camera capture, pose estimation, tracking, normalization, and STG-NF inference on the camera device; stream JPEG frames plus per-frame detection payloads to the supervisor.
- **Rationale**: This matches the feature spec, preserves privacy by avoiding raw camera upload to a backend, keeps the supervisor lightweight, and aligns with the existing app’s camera/inference modules.
- **Alternatives considered**:
  - Stream raw frames to the supervisor and infer there: rejected because it increases bandwidth and couples inference to the monitoring device.
  - Duplicate inference on both devices: rejected due to resource waste and state divergence risk.

## Decision 5: Use a dedicated networking layer with authenticated pairing and framed streaming

- **Decision**: Add `Networking/` services for QR payload generation/parsing, token-authenticated handshake, heartbeat monitoring, and a framed stream protocol over `NWConnection`/`NWListener`.
- **Rationale**: Networking logic does not belong in views or ViewModels under the project’s MVVM rules. Service-layer isolation also makes loopback/integration testing possible and supports future transport upgrades.
- **Alternatives considered**:
  - Put connection logic in `OnboardingViewModel` or `SupervisorViewModel`: rejected because it violates MVVM/service separation.
  - Use Bonjour discovery instead of QR in v1: rejected because QR pairing is the specified user flow.

## Decision 6: Authenticate v1 pairing with a single-use token bound to the pairing-screen lifecycle

- **Decision**: Encode LAN address, port, and a single-use token in the QR payload; require the token in the JSON handshake; invalidate it when the pairing screen disappears; reject reuse.
- **Rationale**: This resolves the biggest security ambiguity in the spec with minimal implementation overhead and directly supports deterministic tests for valid, expired, and reused QR payloads.
- **Alternatives considered**:
  - Trust any device on the LAN: rejected because it leaves pairing unauthenticated.
  - Full certificate/TLS bootstrap in the QR flow: rejected for v1 because it materially increases setup complexity.

## Decision 7: Cap the supervisor at four simultaneous feeds and freeze stale tiles on disconnect

- **Decision**: Support up to four active camera sessions per supervisor and, on disconnect timeout, keep the last frame visible with a stale/disconnected overlay.
- **Rationale**: The four-feed cap gives a concrete test and UI target, while the frozen-frame behavior preserves operator context without implying the feed is live.
- **Alternatives considered**:
  - No explicit feed limit: rejected because it prevents concrete performance and UI planning.
  - Blank the tile immediately on disconnect: rejected because it removes useful operator context.

## Decision 8: Persist threshold locally on each camera device through a persistence service

- **Decision**: Keep the default anomaly threshold at `-1.2`, expose it via a settings surface, and persist the selected value per camera device using `PersistenceService`.
- **Rationale**: This matches the constitution and clarified spec while keeping calibration ownership with the device that performs inference.
- **Alternatives considered**:
  - Reset to default each launch: rejected because it breaks deployment calibration.
  - Let the supervisor push remote thresholds: rejected because that is out of scope for v1 and couples devices unnecessarily.
