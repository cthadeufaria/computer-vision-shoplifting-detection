# iOS Development Guidelines

## Architecture: MVVM

The app follows **Model–View–ViewModel (MVVM)** with a service layer.

```
View  ──▶  ViewModel  ──▶  Service  ──▶  Model
 (SwiftUI)   (@MainActor)   (Actor/class)  (struct/enum)
```

### Layer responsibilities

| Layer | Rules |
|---|---|
| **View** | Renders state, forwards user actions to ViewModel. No business logic, no direct service calls. |
| **ViewModel** | Holds `@Published` state, coordinates services, formats data for display. No `import UIKit`, no AVFoundation/Vision calls. |
| **Service** | Single-responsibility domain logic (camera, pose, scoring). Protocol-backed for testability. |
| **Model** | Pure value types (`struct`, `enum`). `Sendable`. No behavior. |

### Rules

- **No business logic in Views.** If a View contains an `if/else` beyond toggling display, it belongs in the ViewModel.
- **No direct service instantiation in ViewModels.** Inject all dependencies through `init`. The ViewModel must never call `CameraSession()`, `PoseEstimator()`, etc. directly.
- **Error state lives in the ViewModel**, not as `@State` in the View.
- **No singletons** except at the app-composition root (`ShopliftDetectApp`). Pass dependencies down via `init` or SwiftUI environment.
- **No `@AppStorage` in ViewModels.** Abstract UserDefaults behind a `PersistenceService` protocol so tests can inject a mock.

### Dependency injection pattern

```swift
// ✅ Correct
final class DetectionViewModel: ObservableObject {
    init(
        camera: CameraSessionProtocol = CameraSession(),
        estimator: PoseEstimatorProtocol = PoseEstimator(),
        scorer: AnomályScorerProtocol = AnomalyScorer()
    ) { ... }
}

// ❌ Wrong
final class DetectionViewModel: ObservableObject {
    private let camera = CameraSession()      // hardcoded, untestable
}
```

---

## Test-Driven Development

**Every behaviour must have a test before or alongside the code.** Tests are not optional.

**All tests must pass after every code edit.**

- **Before every commit** — run the unit test suite (`ShopliftDetectTests`). A commit must not be made if any unit test is failing.
- **Before every PR** — run the full suite including UI tests (`ShopliftDetectUITests`). A PR must not be merged if any test is failing.

```bash
# Unit tests only (run before each commit)
xcodebuild test -project ShopliftDetect.xcodeproj -scheme ShopliftDetect \
  -destination 'platform=iOS Simulator,name=iPhone 17' \
  -only-testing:ShopliftDetectTests

# Full suite including UI tests (run before each PR)
xcodebuild test -project ShopliftDetect.xcodeproj -scheme ShopliftDetect \
  -destination 'platform=iOS Simulator,name=iPhone 17'
```

### Coverage requirements

| Layer | Required tests |
|---|---|
| ViewModel | All state transitions, all public methods, all error paths |
| Service | All public API paths, including failure/edge cases |
| Model | All computed properties, all custom init logic |
| View | UI tests for every user-visible screen and navigation flow |

### TDD cycle

1. **Red** — write a failing test that describes the behaviour
2. **Green** — write the minimum code to make it pass
3. **Refactor** — clean up without breaking tests

### Test file naming

```
ShopliftDetectTests/
├── Detection/
│   ├── DetectionViewModelTests.swift
│   └── AnomalyScorerTests.swift
├── Camera/
│   └── CameraSessionTests.swift
├── Onboarding/
│   └── OnboardingViewModelTests.swift
└── Mocks/
    ├── MockCameraSession.swift
    ├── MockPoseEstimator.swift
    └── MockAnomalyScorer.swift

ShopliftDetectUITests/
├── DetectionToggleUITests.swift
├── OnboardingUITests.swift
└── PosePreviewUITests.swift
```

### Writing testable services

Every service that talks to hardware, the filesystem, or the network must have a protocol:

```swift
protocol CameraSessionProtocol {
    var framePublisher: AnyPublisher<CMSampleBuffer, Never> { get }
    func start() throws
    func stop()
}

// Production
final class CameraSession: CameraSessionProtocol { ... }

// Test double
final class MockCameraSession: CameraSessionProtocol {
    var startCallCount = 0
    func start() throws { startCallCount += 1 }
    ...
}
```

### ViewModel test pattern

```swift
@MainActor
final class DetectionViewModelTests: XCTestCase {
    var sut: DetectionViewModel!
    var mockCamera: MockCameraSession!

    override func setUp() {
        mockCamera = MockCameraSession()
        sut = DetectionViewModel(camera: mockCamera)
    }

    func test_start_setsStateToRunning() throws {
        try sut.start()
        XCTAssertEqual(sut.state, .running)
    }

    func test_start_whenCameraFails_setsError() throws {
        mockCamera.shouldThrowOnStart = true
        XCTAssertThrowsError(try sut.start())
        XCTAssertNotNil(sut.errorMessage)
    }
}
```

### UI test pattern

Use `accessibilityIdentifier` (already set on key controls) to locate elements. Never rely on display text for element lookup — it breaks on localisation.

```swift
func test_startDetection_showsDetectionScreen() {
    app.buttons["startDetectionButton"].tap()
    XCTAssertTrue(app.buttons["dismissDetectionButton"].waitForExistence(timeout: 2))
}
```

---

## Known violations to fix

These exist in the current codebase and must be resolved before adding new features:

| File | Violation | Fix |
|---|---|---|
| `DetectionViewModel` | Hardcodes `CameraSession`, `PoseEstimator`, `KeypointConverter`, `AnomalyScorer` | Inject via `init` with protocol types |
| `DetectionViewModel` | IoU / track-matching logic in ViewModel | Extract to `TrackingService` |
| `PosePreviewViewModel` | Hardcodes `CameraSession`, `PoseEstimator` | Inject via `init` |
| `OnboardingViewModel` | Direct `@AppStorage` and `AVCaptureDevice` calls | Extract to `PersistenceService` and `PermissionService` |
| `AnomalyScorer` | Hardcoded threshold `-1.2` | Accept as `init(threshold:)` parameter |
| `AppEnvironment` | Empty singleton | Implement or delete |

---

## SwiftUI-specific rules

- Prefer `@StateObject` in the View that owns the ViewModel; pass as `@ObservedObject` to children.
- Use `task {}` modifier for async ViewModel setup rather than `.onAppear` + `Task {}`.
- Never put `async` work directly in `body`.
- Keep `body` short — extract subviews into separate `View` structs when a component exceeds ~40 lines.

## Concurrency

- Mark ViewModels `@MainActor` at the class level, not method by method.
- Services that do off-thread work should be `actor` (see `FrameBuffer`) or return on `MainActor` via `await`.
- Never call `DispatchQueue.main.async` — use `await MainActor.run` or `@MainActor` annotations.

## Recent Changes
- 001-camera-supervisor-p2p: Added Swift 6.0 for the app, Python 3 for fixture/model conversion and reporting scripts + SwiftUI, Combine, AVFoundation, Vision, CoreML, Network.framework with TLS parameters, CoreImage, XCTest, XCUITest, XcodeGen, SwiftLin
- 001-camera-supervisor-p2p: Added Swift 6.0 for the app, Python 3 for fixture/model conversion scripts + SwiftUI, Combine, AVFoundation, Vision, CoreML, Network.framework, CoreImage, XCTest, XCUITest, XcodeGen
- 001-camera-supervisor-p2p: Added [if applicable, e.g., PostgreSQL, CoreData, files or N/A]

## Active Technologies
- Swift 6.0 for the app, Python 3 for fixture/model conversion and reporting scripts + SwiftUI, Combine, AVFoundation, Vision, CoreML, Network.framework with TLS parameters, CoreImage, XCTest, XCUITest, XcodeGen, SwiftLin (001-camera-supervisor-p2p)
- UserDefaults via `PersistenceService`; bundled `.mlpackage` and JSON fixtures in app/test resources; markdown planning artifacts in `specs/`; no database (001-camera-supervisor-p2p)
