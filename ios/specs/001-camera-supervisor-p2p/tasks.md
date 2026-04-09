# Tasks: iOS Smart Camera to Supervisory Device P2P App

**Input**: Design documents from `/Users/bernese/git/computer-vision-shoplifting-detection/ios/specs/001-camera-supervisor-p2p/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Tests are mandatory for this feature because the spec, constitution, and repo instructions require TDD for every behavior.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g. `US1`, `US2`, `US3`, `US4`)
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Align the existing iOS project with the plan and constitution before feature work starts.

- [X] T001 Update iOS deployment targets and strict-concurrency project settings in /Users/bernese/git/computer-vision-shoplifting-detection/ios/project.yml and /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect.xcodeproj/project.pbxproj
- [X] T002 [P] Update CoreML conversion assumptions and model-source comments for the Apr01_1416 checkpoint in /Users/bernese/git/computer-vision-shoplifting-detection/ios/scripts/convert_stgnf_to_coreml.py
- [X] T003 [P] Regenerate the Xcode project from /Users/bernese/git/computer-vision-shoplifting-detection/ios/project.yml and verify resources/tests remain wired in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect.xcodeproj/project.pbxproj

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared architecture and protocols that block all stories.

**⚠️ CRITICAL**: No user story work should begin until this phase is complete.

- [X] T004 [P] Add shared networking/domain model types in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Networking/DeviceRole.swift, /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Networking/PairingSession.swift, /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Networking/PairingToken.swift, /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Networking/ConnectionState.swift, /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Networking/VideoFrame.swift, and /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Networking/DetectionResult.swift
- [X] T005 [P] Add protocol abstractions for networking and settings in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Core/Protocols/PairingServiceProtocol.swift, /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Core/Protocols/StreamingServiceProtocol.swift, and /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Core/Protocols/SettingsServiceProtocol.swift
- [X] T006 [P] Extend persistence and permission services for role persistence, threshold persistence, and QR-scanner permissions in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Core/Services/PersistenceService.swift and /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Core/Services/PermissionService.swift
- [X] T007 [P] Add shared mocks for networking and settings dependencies in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetectTests/Mocks/MockPairingService.swift, /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetectTests/Mocks/MockStreamingService.swift, and /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetectTests/Mocks/MockSettingsService.swift
- [X] T008 Refactor app composition to inject services through the root environment in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/App/AppEnvironment.swift and /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/App/ShopliftDetectApp.swift
- [X] T009 Refactor existing view models to remove remaining hard-coded dependencies in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Detection/DetectionViewModel.swift, /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Detection/PosePreviewViewModel.swift, /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Onboarding/OnboardingViewModel.swift, and /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Home/HomeViewModel.swift

**Checkpoint**: Foundation ready. User story work can begin.

---

## Phase 3: User Story 1 - Camera Device Detects and Streams in Real Time (Priority: P1) 🎯 MVP

**Goal**: Deliver camera-role onboarding-to-detection flow with real-time pose inference, warmup, multi-person tracking, score cards, threshold persistence, and outbound stream publishing hooks.

**Independent Test**: Launch the app on a physical device, complete onboarding as Smart Camera, tap Start Detection, and confirm skeleton overlay appears within 1 second while score cards move from warmup to GOOD/ANOMALY after ~0.8 seconds.

### Tests for User Story 1 ⚠️

- [ ] T010 [P] [US1] Add camera detection state-transition tests in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetectTests/Detection/DetectionViewModelTests.swift
- [ ] T011 [P] [US1] Add tracking edge-case tests for multi-person IoU matching in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetectTests/Detection/TrackingServiceTests.swift
- [ ] T012 [P] [US1] Add threshold persistence and scoring boundary tests in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetectTests/Model/AnomalyScorerTests.swift
- [ ] T013 [P] [US1] Add pose normalization and fixture parity coverage for the latest model input contract in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetectTests/Pose/PoseNormalizerTests.swift and /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetectTests/Model/STGNFModelIntegrationTests.swift
- [ ] T014 [P] [US1] Add camera-role UI flow coverage for start/dismiss/warmup behavior in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetectUITests/DetectionToggleUITests.swift

### Implementation for User Story 1

- [ ] T015 [P] [US1] Add persisted detection settings model and service wiring in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Core/Domain/DetectionSettings.swift and /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Core/Services/PersistenceService.swift
- [ ] T016 [P] [US1] Update anomaly scoring and model wrapper for configurable threshold and Apr01_1416 model assumptions in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Core/Model/AnomalyScorer.swift and /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Core/Model/STGNFModelWrapper.swift
- [ ] T017 [P] [US1] Finalize tracking and rolling-window infrastructure in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Detection/TrackingService.swift and /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Core/Model/FrameBuffer.swift
- [ ] T018 [US1] Implement camera detection orchestration with injected services and outbound streaming callbacks in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Detection/DetectionViewModel.swift
- [ ] T019 [US1] Update camera-role UI for warmup, score cards, streaming indicator, and threshold settings in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Detection/DetectionView.swift, /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Detection/DetectionViewComponents.swift, /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Detection/ScoreCardView.swift, and /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Home/HomeView.swift

**Checkpoint**: User Story 1 should be fully functional and independently testable as the MVP.

---

## Phase 4: User Story 3 - Device Pairing via QR Code (Priority: P2)

**Goal**: Deliver authenticated QR-based pairing with single-use token lifecycle, JSON handshake validation, and heartbeat/disconnect detection.

**Independent Test**: Display QR on the camera device, scan it on the supervisor device, complete the handshake, and confirm both devices show connected state; invalid or reused tokens must be rejected and scanning must recover.

### Tests for User Story 3 ⚠️

- [ ] T020 [P] [US3] Add QR payload parsing and token lifecycle tests in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetectTests/Networking/PairingServiceTests.swift
- [ ] T021 [P] [US3] Add framed handshake and heartbeat timeout tests in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetectTests/Networking/StreamProtocolTests.swift
- [ ] T022 [P] [US3] Add onboarding QR display/scan and invalid-token UI tests in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetectUITests/OnboardingUITests.swift and /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetectUITests/RoleSelectionUITests.swift

### Implementation for User Story 3

- [ ] T023 [P] [US3] Implement framed stream transport and heartbeat message handling in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Networking/StreamProtocol.swift and /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Networking/StreamingService.swift
- [ ] T024 [P] [US3] Implement pairing listener/connector, QR payload generation, token validation, and handshake messages in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Networking/PairingService.swift
- [ ] T025 [US3] Update onboarding view model for role-based QR presentation, scanning, connection state, and rescan recovery in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Onboarding/OnboardingViewModel.swift
- [ ] T026 [US3] Build camera QR display and supervisor QR scanner screens with connection feedback in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Onboarding/QRCodeDisplayView.swift, /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Onboarding/QRScannerView.swift, and /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Onboarding/OnboardingView.swift
- [ ] T027 [US3] Route pairing state into the role-appropriate home flow in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Home/HomeViewModel.swift and /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Home/HomeView.swift

**Checkpoint**: User Story 3 should be fully functional and independently testable.

---

## Phase 5: User Story 2 - Supervisor Device Monitors Multiple Camera Feeds (Priority: P2)

**Goal**: Deliver a supervisor grid that monitors up to four camera feeds, updates each tile independently, supports full-screen drill-in, and preserves stale frames on disconnect.

**Independent Test**: Pair a supervisor with one or more camera devices on the same Wi-Fi network and confirm live tiles appear within 2 seconds, anomaly badges propagate within 500 ms, and a disconnected feed freezes the last frame with a stale overlay.

### Tests for User Story 2 ⚠️

- [ ] T028 [P] [US2] Add supervisor session-limit and stale-tile state tests in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetectTests/Networking/PairingServiceTests.swift and /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetectTests/Detection/DetectionViewModelTests.swift
- [ ] T029 [P] [US2] Add supervisor view-model tests for tile updates, full-screen selection, and disconnect behavior in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetectTests/Supervisor/SupervisorViewModelTests.swift
- [ ] T030 [P] [US2] Add supervisor monitoring UI tests for empty grid, live tile, full-screen expansion, and fifth-camera rejection in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetectUITests/SupervisorMonitoringUITests.swift

### Implementation for User Story 2

- [ ] T031 [P] [US2] Add supervisor feed-grid domain types in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Supervisor/SupervisorFeedTileState.swift and /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Supervisor/SupervisorFeedGrid.swift
- [ ] T032 [US2] Implement supervisor session orchestration, tile updates, session limits, and stale-frame behavior in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Supervisor/SupervisorViewModel.swift
- [ ] T033 [US2] Build supervisor monitoring UI, tile thumbnails, stale overlay, and full-screen feed detail in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Supervisor/SupervisorView.swift, /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Supervisor/DeviceRowView.swift, /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Supervisor/CameraFeedDetailView.swift, and /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Home/SupervisorHomeView.swift
- [ ] T034 [US2] Connect camera-side stream publishing and supervisor-side stream consumption in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Detection/DetectionViewModel.swift, /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Networking/StreamingService.swift, and /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Supervisor/SupervisorViewModel.swift

**Checkpoint**: User Stories 2 and 3 should now work independently and together.

---

## Phase 6: User Story 4 - Onboarding and Role Persistence (Priority: P3)

**Goal**: Deliver the four-page onboarding flow, role persistence, permission handling, and relaunch routing to the correct home screen.

**Independent Test**: Complete onboarding, terminate the app, relaunch, and confirm onboarding is skipped and the correct role-specific home screen appears immediately.

### Tests for User Story 4 ⚠️

- [X] T035 [P] [US4] Add onboarding view-model tests for page progression, role confirmation, persistence writes, and permission denial handling in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetectTests/Onboarding/OnboardingViewModelTests.swift
- [X] T036 [P] [US4] Add home routing tests for persisted-role launch behavior in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetectTests/Home/HomeViewModelTests.swift
- [X] T037 [P] [US4] Add full onboarding and relaunch UI coverage in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetectUITests/OnboardingUITests.swift

### Implementation for User Story 4

- [X] T038 [P] [US4] Finalize onboarding page models and role-selection UI in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Onboarding/OnboardingPageView.swift and /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Onboarding/RoleSelectionView.swift
- [X] T039 [US4] Implement onboarding flow state, persisted role writes, and permission CTA handling in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Onboarding/OnboardingViewModel.swift and /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Onboarding/OnboardingView.swift
- [X] T040 [US4] Implement launch routing for persisted role and onboarding-complete state in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Home/HomeViewModel.swift and /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Home/HomeView.swift

**Checkpoint**: All user stories should now be independently functional.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Final hardening, docs alignment, and validation across stories.

- [ ] T041 [P] Add or update Info.plist usage strings and local-network privacy text in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Resources/Info.plist
- [ ] T042 [P] Add missing fixture or helper coverage for networking and supervisor flows in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetectTests/Fixtures and /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetectTests/Mocks
- [ ] T043 Run unit and UI test suites from /Users/bernese/git/computer-vision-shoplifting-detection/ios/quickstart.md and fix regressions across /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect, /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetectTests, and /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetectUITests
- [ ] T044 Run two-device manual validation from /Users/bernese/git/computer-vision-shoplifting-detection/ios/specs/001-camera-supervisor-p2p/quickstart.md and document any follow-up notes in /Users/bernese/git/computer-vision-shoplifting-detection/ios/specs/001-camera-supervisor-p2p/quickstart.md

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1: Setup**: No dependencies.
- **Phase 2: Foundational**: Depends on Phase 1 and blocks all story work.
- **Phase 3: US1**: Depends on Phase 2 only and is the MVP.
- **Phase 4: US3**: Depends on Phase 2 and can proceed after US1 if working sequentially.
- **Phase 5: US2**: Depends on Phase 2 and on US3 pairing/stream contracts for full end-to-end behavior.
- **Phase 6: US4**: Depends on Phase 2 and can proceed in parallel with later stories once the shared services are stable.
- **Phase 7: Polish**: Depends on all desired user stories being complete.

### User Story Dependencies

- **US1 (P1)**: No dependency on other user stories after foundational work.
- **US3 (P2)**: No dependency on other user stories after foundational work.
- **US2 (P2)**: Depends on US3 for authenticated pairing and live stream sessions, but remains independently testable once those services exist.
- **US4 (P3)**: No hard dependency on other user stories after foundational work.

### Within Each User Story

- Test tasks must be written and fail before implementation tasks begin.
- Domain and protocol types should land before service orchestration.
- ViewModels should land before UI integration.
- Story checkpoints should be validated before moving to the next phase in a single-developer flow.

### Parallel Opportunities

- `T002` and `T003` can run in parallel after `T001`.
- `T004` to `T007` can run in parallel during foundational work.
- US1 test tasks `T010` to `T014` can run in parallel.
- US3 test tasks `T020` to `T022` can run in parallel.
- US2 test tasks `T028` to `T030` can run in parallel.
- US4 test tasks `T035` to `T037` can run in parallel.
- US4 can proceed in parallel with US3 once foundational dependency injection is done.

---

## Parallel Example: User Story 1

```bash
Task: "Add camera detection state-transition tests in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetectTests/Detection/DetectionViewModelTests.swift"
Task: "Add tracking edge-case tests for multi-person IoU matching in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetectTests/Detection/TrackingServiceTests.swift"
Task: "Add threshold persistence and scoring boundary tests in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetectTests/Model/AnomalyScorerTests.swift"
```

## Parallel Example: User Story 3

```bash
Task: "Add QR payload parsing and token lifecycle tests in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetectTests/Networking/PairingServiceTests.swift"
Task: "Add framed handshake and heartbeat timeout tests in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetectTests/Networking/StreamProtocolTests.swift"
Task: "Add onboarding QR display/scan and invalid-token UI tests in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetectUITests/OnboardingUITests.swift and /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetectUITests/RoleSelectionUITests.swift"
```

## Parallel Example: User Story 4

```bash
Task: "Add onboarding view-model tests for page progression, role confirmation, persistence writes, and permission denial handling in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetectTests/Onboarding/OnboardingViewModelTests.swift"
Task: "Add home routing tests for persisted-role launch behavior in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetectTests/Home/HomeViewModelTests.swift"
Task: "Add full onboarding and relaunch UI coverage in /Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetectUITests/OnboardingUITests.swift"
```

## Implementation Strategy

### MVP First

1. Complete Setup and Foundational phases.
2. Complete US1.
3. Validate detection MVP on device before starting networking-heavy work.

### Incremental Delivery

1. Add US1 for camera detection MVP.
2. Add US3 for authenticated pairing.
3. Add US2 for supervisor monitoring.
4. Add US4 for polished onboarding/routing persistence.
5. Finish with cross-cutting validation and regression fixes.

### Parallel Team Strategy

1. One developer handles Setup and Foundational tasks.
2. After Phase 2:
   - Developer A: US1
   - Developer B: US3
   - Developer C: US4
3. Once US3 stabilizes, a developer takes US2 on top of the shared networking contracts.

## Notes

- All tasks use the required checklist format.
- Story phases are structured for TDD: tests first, implementation second.
- US2 is intentionally placed after US3 because pairing and framed transport are prerequisites for realistic supervisor monitoring.
