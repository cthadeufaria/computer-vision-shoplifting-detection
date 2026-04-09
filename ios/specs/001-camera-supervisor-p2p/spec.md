# Feature Specification: iOS Smart Camera to Supervisory Device P2P App

**Feature Branch**: `001-camera-supervisor-p2p`
**Created**: 2026-04-08
**Status**: Draft
**Input**: User description: "an app where one iOS device working as a smart camera device connects to another iOS device that works as a supervisory device"

---

## Clarifications

### Session 2026-04-09

- Q: How should v1 authenticate a supervisor during pairing on the local LAN? → A: Require a single-use pairing token in the QR/handshake.
- Q: How many simultaneous camera feeds should one supervisor support in v1? → A: Up to 4 simultaneous camera feeds.
- Q: How long should the QR code and pairing token remain valid in v1? → A: Only while the pairing screen is visible.
- Q: How should the supervisor tile behave when a camera feed disconnects or stops updating? → A: Freeze the last frame and show a stale/disconnected overlay.
- Q: How should the anomaly threshold setting persist in v1? → A: Persist it locally per camera device.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Camera Device Detects and Streams in Real Time (Priority: P1)

A security operator places an iPhone near a retail area and opens the app. They select
the "Smart Camera" role during onboarding, then tap "Start Detection" on the home screen.
The device runs pose-based anomaly detection continuously on the live camera feed and
streams annotated frames plus detection results to any paired supervisory device over
the local Wi-Fi network.

**Why this priority**: This is the core value proposition. Without a functioning
detection pipeline no other feature is meaningful.

**Independent Test**: Launch the app on a physical device, complete onboarding as
"Smart Camera", tap Start Detection — skeleton overlay appears within 1 second and
score cards show "Warming up" then GOOD/ANOMALY labels after ~0.8 s.

**Acceptance Scenarios**:

1. **Given** the app is launched for the first time, **When** the user selects the
   Smart Camera role, **Then** a QR code is displayed for pairing and onboarding
   advances to the camera permission page.
2. **Given** the user has granted camera permission and tapped Start Detection,
   **When** a person walks in front of the camera, **Then** a skeleton overlay appears
   on the person within one second.
3. **Given** the warmup period (first 24 frames) is in progress, **When** the score
   card is visible, **Then** it shows "Warming up X/24" in gray.
4. **Given** the warmup is complete, **When** a pose window is inferred, **Then** the
   score card shows either GOOD (green) or ANOMALY (red) with a numeric score.
5. **Given** a supervisor is connected, **When** the camera device is detecting,
   **Then** a green streaming indicator is visible on the detection screen.

---

### User Story 2 — Supervisor Device Monitors Multiple Camera Feeds (Priority: P2)

A security manager opens the app on an iPad, selects "Supervisory View" during
onboarding, and scans the QR code displayed on a camera device. The iPad shows a
live grid of up to four connected camera feeds, each tile displaying the incoming video
frames and anomaly score overlays in near-real-time.

**Why this priority**: The multi-device supervisory view is the key differentiator
over a single-device solution — it is the delivery mechanism for alerts to a
monitoring station.

**Independent Test**: Connect two physical devices on the same Wi-Fi. Camera device
shows QR → iPad supervisor scans it → both show "Connected" → iPad displays live
feed tile with score overlay from the camera device.

**Acceptance Scenarios**:

1. **Given** the supervisor selects the Supervisory View role, **When** onboarding
   completes, **Then** the home screen shows a camera-feed grid (initially empty/placeholder).
2. **Given** the supervisor scans a camera device's QR code, **When** pairing
   succeeds, **Then** a live tile for that camera appears in the grid within 2 seconds.
3. **Given** a connected camera tile is visible, **When** the camera device detects an
   anomaly, **Then** the supervisor tile shows a red ANOMALY badge within 500 ms.
4. **Given** up to four camera devices are paired, **When** any device sends a frame,
   **Then** each tile updates independently without affecting other tiles.
5. **Given** four camera devices are already connected, **When** the supervisor scans a
   fifth camera QR code, **Then** the app blocks the connection attempt and explains that
   v1 supports a maximum of four simultaneous feeds.
6. **Given** a camera tile is tapped, **When** the supervisor interaction occurs,
   **Then** the tile expands to full-screen with the same score overlay layout as the
   camera device's detection view.
7. **Given** a connected camera stops sending frames, **When** the disconnect timeout is
   reached, **Then** the supervisor freezes the last received frame and displays a
   stale/disconnected overlay on that tile.

---

### User Story 3 — Device Pairing via QR Code (Priority: P2)

The camera device displays a QR code encoding its local IP address and port. The
supervisory device's onboarding screen opens a QR scanner that reads this code and
initiates a TCP connection. The QR payload also carries a single-use pairing token
that the supervisor must present during the JSON handshake before the camera accepts
the session. The QR code and token remain valid only while the pairing screen is visible,
and a fresh token is generated each time that screen appears. Both devices show a
"Connected" confirmation and role routing takes effect.

**Why this priority**: Pairing is a prerequisite for the supervisory use case (P2).
It is grouped at the same priority because the two stories are co-dependent.

**Independent Test**: Camera device shows QR → supervisor scans it → both devices
show connection status "Connected" → role-appropriate home screen shown on each.

**Acceptance Scenarios**:

1. **Given** the camera role is selected, **When** the QR code screen is shown,
   **Then** the QR code encodes a payload matching `sdlink://<LAN_IP>:<PORT>`.
2. **Given** the supervisor role is selected, **When** the user points the camera at
   the QR code, **Then** the payload is parsed and a TCP connection is attempted.
3. **Given** the QR payload is invalid (wrong scheme, non-LAN IP, malformed),
   **When** scanning occurs, **Then** an error message is shown and scanning resumes.
4. **Given** a valid QR payload is scanned, **When** the supervisor presents the
   included single-use pairing token during the handshake, **Then** the camera accepts
   only that token and rejects missing, reused, or mismatched tokens.
5. **Given** a successful connection is established, **When** the handshake completes,
   **Then** both devices show the role-appropriate home screen with a "Connected" indicator.
6. **Given** the camera leaves the pairing screen, **When** a supervisor later tries to
   reuse the previously scanned QR payload, **Then** the handshake is rejected and the
   camera requires a newly displayed QR code.
7. **Given** the camera device loses connectivity or goes to background,
   **When** a heartbeat is missed, **Then** the supervisor tile shows a disconnected
   state within 5 seconds.

---

### User Story 4 — Onboarding and Role Persistence (Priority: P3)

A new user opens the app and is guided through a 4-page onboarding flow: Welcome,
Role Selection, How It Works, and Camera Permission. The selected role is persisted
so subsequent launches bypass onboarding and route directly to the correct home screen.

**Why this priority**: Onboarding is a one-time setup path; it does not affect
daily operational use after the first launch.

**Independent Test**: Complete onboarding → force-quit app → re-launch → onboarding
is skipped and the correct role home screen is shown immediately.

**Acceptance Scenarios**:

1. **Given** the app is launched for the first time, **When** the welcome screen
   appears, **Then** a "Next" button advances to the Role Selection page.
2. **Given** the role selection page is shown, **When** the user taps a role card,
   **Then** the role is highlighted and a "Confirm" action becomes available.
3. **Given** onboarding is completed, **When** the app is relaunched, **Then**
   onboarding is skipped and the role-appropriate home screen is shown directly.
4. **Given** the user is on the camera permission page, **When** permission is
   granted, **Then** onboarding completes and the home screen is shown.

---

### Edge Cases

- What happens when two devices are on different Wi-Fi networks?
  → Pairing fails; the app shows "Connect both devices to the same Wi-Fi network."
- What happens when a supervisor presents a missing, reused, or invalid pairing token?
  → The handshake is rejected, scanning remains available, and the user is prompted to rescan the QR code.
- What happens when the supervisor scans a QR code but the camera leaves the pairing screen before handshake completes?
  → The token expires immediately, the handshake is rejected, and the supervisor must scan a newly displayed QR code.
- What happens when the camera device's LAN IP changes mid-session?
  → The TCP connection drops; the supervisor tile shows disconnected state and
    the user must re-pair.
- What happens when a supervisor attempts to pair a fifth camera device?
  → The pairing attempt is refused and the UI explains that v1 supports up to four simultaneous feeds.
- What happens when a connected camera stops sending frames unexpectedly?
  → The supervisor keeps the last received frame visible and overlays a stale/disconnected status within 5 seconds.
- What happens when the anomaly threshold produces false positives on normal retail footage?
  → The threshold is adjustable in a Settings sheet, defaults to -1.2, and is
    persisted locally on each camera device. A note on the detection screen explains
    calibration may be needed.
- What happens when camera permission is denied?
  → A persistent prompt directs the user to Settings; the Start Detection button
    is disabled until permission is granted.
- What happens on the first 24 frames (warmup period)?
  → Score cards show "Warming up X/24"; no GOOD/ANOMALY label is displayed until
    the first full 24-frame window is accumulated.
- What happens when multiple people are in frame simultaneously?
  → Each tracked person gets an independent score card; tracking uses IoU-based
    identity association.

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The app MUST offer a first-launch onboarding flow with role selection
  (Smart Camera or Supervisory View).
- **FR-002**: The selected device role MUST be persisted across app launches so
  onboarding is shown only once.
- **FR-003**: The Smart Camera role MUST run a real-time pose anomaly detection
  pipeline on live camera frames and display GOOD/ANOMALY labels with a skeleton overlay.
- **FR-004**: The anomaly score threshold MUST default to -1.2 and MUST be adjustable
  by the user through a Settings sheet without a code change, with the selected value
  persisted locally per camera device across app launches.
- **FR-005**: The Smart Camera device MUST display a QR code encoding its local LAN
  IP address, TCP port, and a single-use pairing token for pairing.
- **FR-006**: The Supervisory device MUST include a QR scanner that parses the pairing
  payload and initiates a TCP connection to the camera device.
- **FR-007**: Both devices MUST exchange a JSON handshake after connecting that
  identifies role, device name, and the single-use pairing token.
- **FR-008**: The camera device MUST stream annotated JPEG frames and detection results
  to the connected supervisor at approximately 10 fps.
- **FR-009**: The Supervisory device MUST display a live grid of connected camera feeds,
  each showing incoming frames and score overlays, with support for up to four simultaneous feeds in v1.
- **FR-010**: Tapping a supervisor feed tile MUST expand it to full-screen with the
  same overlay layout as the camera device's detection view.
- **FR-011**: The system MUST detect and display a disconnected state on the supervisor
  tile within 5 seconds of a lost connection by freezing the last received frame and
  showing a stale/disconnected overlay.
- **FR-012**: The camera device MUST display a streaming status indicator
  (connected / standalone) on the detection screen.
- **FR-013**: All video transmission MUST occur over local LAN only; no internet
  routing is permitted in the v1 implementation.
- **FR-014**: The detection pipeline MUST support multiple simultaneous people in frame,
  each with an independent score card.
- **FR-015**: The camera device MUST reject handshakes with missing, invalid, expired,
  or previously used pairing tokens.
- **FR-016**: The Supervisory device MUST reject or block pairing attempts that would
  exceed four simultaneous connected camera feeds.
- **FR-017**: The camera device MUST invalidate the displayed QR code and pairing token
  immediately when the pairing screen is dismissed or replaced, and MUST generate a fresh
  token the next time the pairing screen is shown.

### Key Entities

- **DeviceRole**: `camera` or `supervisor`; persisted in user preferences.
- **DetectionSettings**: Camera-local persisted configuration including anomaly threshold.
- **PairingSession**: One active peer connection per device; observable connection state and
  single-use pairing token lifecycle.
- **PairingToken**: A single-use credential embedded in the QR payload; valid only while the
  camera pairing screen is visible.
- **SupervisorFeedGrid**: A collection of up to four active camera feed tiles, each with
  connection state, latest frame, and latest anomaly overlay.
- **SupervisorFeedTileState**: Per-tile display state including active, stale/disconnected,
  latest received frame, and latest anomaly overlay.
- **AnomalyResult**: Score (Float), label (normal / anomaly / warmup), timestamp.
- **DetectionState**: idle / warmingUp(frames collected, needed) / running(latest result) / error.
- **VideoFrame**: JPEG-compressed frame bytes plus 8-byte timestamp; transmitted at 10 fps.
- **DetectionResult**: Per-person track ID, score, label, and keypoints; transmitted alongside video frames.
- **PoseSkeleton**: 18 keypoints in OpenPose COCO18 order with pixel coordinates and confidence.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user can complete onboarding and start detection within 60 seconds of
  first launch on a supported device.
- **SC-002**: Skeleton overlay appears on a detected person within 1 second of the
  person entering the camera frame.
- **SC-003**: After the 24-frame warmup period (~0.8 s at 30 fps), GOOD/ANOMALY labels
  update at least once per second.
- **SC-004**: Two devices can be paired in under 30 seconds using the QR code flow on
  a shared Wi-Fi network.
- **SC-005**: Supervisor feed tiles update at 10 fps or better from each connected
  camera device under normal LAN conditions with four simultaneous feeds active.
- **SC-006**: An anomaly event on the camera device appears on the supervisor tile within
  500 ms of detection.
- **SC-007**: The app runs at 30 fps camera capture on an iPhone 8 (iOS 15) without
  sustained frame drops or thermal throttling during a 10-minute session.
- **SC-008**: All 43 unit, integration, and UI tests pass before the feature is
  considered complete.
- **SC-009**: Pose normalization output matches the Python reference implementation
  within a tolerance of 1e-5 per element on the seeded fixture dataset.

---

## Assumptions

- Both devices MUST be on the same local Wi-Fi network; Bluetooth and internet-routed
  pairing are out of scope for v1.
- The CoreML model (STGNFModel.mlpackage) is bundled with the app; no on-device
  training or model download is required.
- The minimum supported iOS version is 15.0; all API usage MUST be compatible with iOS 15.
- The app is a proof-of-concept; App Store submission and MDM distribution are out of
  scope for v1.
- The ShanghaiTech-trained model may produce false positives on normal retail footage;
  this is expected and documented in the UI with a threshold calibration note.
- The anomaly threshold is configured and stored independently on each camera device;
  supervisory devices do not change camera thresholds in v1.
- The supervisory device requires camera permission only for QR code scanning, not for
  inference.
- A single camera device can be paired to one supervisor at a time in v1; one-to-many
  supervisor views are handled by the supervisor side managing up to four simultaneous sessions.
- A future Rust compute kernel is a post-POC concern; the entire v1 app is written in Swift.
