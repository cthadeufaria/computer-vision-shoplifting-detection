# Data Model: iOS Smart Camera to Supervisory Device P2P App

## Entities

### DeviceRole

- **Type**: enum
- **Values**: `camera`, `supervisor`
- **Persistence**: Stored in `PersistenceService` after onboarding completion.
- **Validation**: Must be selected before onboarding can complete.

### DetectionSettings

- **Type**: struct
- **Fields**:
  - `anomalyThreshold: Float`
- **Validation**:
  - Defaults to `-1.2`
  - Must be persisted locally per camera device
- **Relationships**:
  - Owned by one camera-role installation

### PairingToken

- **Type**: struct/value object
- **Fields**:
  - `value: String`
  - `issuedAt: Date`
  - `isConsumed: Bool`
  - `isVisibleOnScreen: Bool`
- **Validation**:
  - Must be unique per pairing-screen presentation
  - Must be rejected if missing, reused, invalid, or no longer visible
- **Lifecycle**:
  - `generated` → `presented` → `consumed` or `invalidated`

### PairingSession

- **Type**: struct or class managed by networking service
- **Fields**:
  - `sessionID: UUID`
  - `role: DeviceRole`
  - `deviceName: String`
  - `host: String`
  - `port: UInt16`
  - `connectionState: ConnectionState`
  - `heartbeatDeadline: Date`
  - `token: PairingToken?`
- **Validation**:
  - One active session per camera device
  - Supervisor may hold at most four active sessions
- **Lifecycle**:
  - `idle` → `listening` or `connecting` → `handshaking` → `connected` → `stale`/`disconnected`

### ConnectionState

- **Type**: enum
- **Values**:
  - `idle`
  - `listening`
  - `connecting`
  - `handshaking`
  - `connected`
  - `stale`
  - `disconnected`
  - `failed(message)`

### SupervisorFeedGrid

- **Type**: aggregate
- **Fields**:
  - `tiles: [SupervisorFeedTileState]`
- **Validation**:
  - Maximum count is 4
  - Tiles update independently

### SupervisorFeedTileState

- **Type**: struct
- **Fields**:
  - `cameraID: UUID`
  - `deviceName: String`
  - `connectionState: ConnectionState`
  - `latestFrame: VideoFrame?`
  - `latestResults: [DetectionResult]`
  - `lastUpdatedAt: Date?`
  - `isFullscreenSelected: Bool`
- **Validation**:
  - When state becomes `stale` or `disconnected`, `latestFrame` remains visible until replaced or session removed

### VideoFrame

- **Type**: struct
- **Fields**:
  - `timestamp: UInt64`
  - `jpegData: Data`
  - `width: Int`
  - `height: Int`
- **Validation**:
  - Encoded for supervisor transport at approximately 10 fps
  - Not written to disk

### DetectionResult

- **Type**: struct
- **Fields**:
  - `trackID: Int`
  - `score: Float`
  - `label: DetectionLabel`
  - `keypoints: [Keypoint]`
  - `boundingBox: CGRect`
  - `timestamp: Date`
- **Validation**:
  - One result per tracked person
  - Uses IoU tracking with `iou_threshold=0.3` and `max_missing=6`

### DetectionLabel

- **Type**: enum
- **Values**:
  - `warmup`
  - `normal`
  - `anomaly`

### DetectionState

- **Type**: enum
- **Values**:
  - `idle`
  - `warmingUp(collected: Int, needed: Int)`
  - `running(results: [AnomalyResult])`
  - `error(message: String)`
- **Transitions**:
  - `idle` → `warmingUp`
  - `warmingUp` → `running`
  - any state → `error`
  - `running`/`warmingUp` → `idle` on stop

### AnomalyResult

- **Type**: struct
- **Fields**:
  - `score: Float`
  - `label: DetectionLabel`
  - `timestamp: Date`
- **Validation**:
  - `score == -nll`
  - `score <= threshold` maps to `anomaly`

### PoseSkeleton

- **Type**: struct
- **Fields**:
  - `keypoints: [Keypoint]`
  - `trackID: Int?`
- **Validation**:
  - Exactly 18 keypoints in COCO18/OpenPose order
  - Derived from Vision body-pose output through `KeypointConverter`

### Keypoint

- **Type**: struct
- **Fields**:
  - `x: Float`
  - `y: Float`
  - `confidence: Float`
- **Validation**:
  - Pixel coordinates before normalization

## Relationships

- One app installation owns one persisted `DeviceRole`.
- A camera-role installation owns one `DetectionSettings` value and can expose one active `PairingSession`.
- A supervisor installation owns one `SupervisorFeedGrid` containing up to four `SupervisorFeedTileState` records.
- Each `SupervisorFeedTileState` is backed by one connected `PairingSession`.
- Each `VideoFrame` may be accompanied by zero or more `DetectionResult` records.
- Each `DetectionResult` references one tracked person and one `PoseSkeleton`.

## Derived State and Rules

- The QR payload is derived from `host`, `port`, and current `PairingToken`.
- A token becomes invalid as soon as the camera leaves the pairing screen.
- `SupervisorFeedTileState.connectionState` transitions to `stale` after heartbeat/frame timeout and preserves `latestFrame`.
- Threshold persistence is camera-local; supervisor devices do not mutate remote `DetectionSettings` in v1.
