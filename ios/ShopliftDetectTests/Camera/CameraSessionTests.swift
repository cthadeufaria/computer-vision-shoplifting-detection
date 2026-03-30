import XCTest
@testable import ShopliftDetect

@MainActor
final class CameraSessionTests: XCTestCase {

    // MARK: - CameraError descriptions

    func test_permissionDenied_hasExpectedDescription() {
        XCTAssertEqual(
            CameraError.permissionDenied.errorDescription,
            "Camera access is required. Enable it in Settings → ShopliftDetect → Camera."
        )
    }

    func test_deviceUnavailable_hasDescription() {
        XCTAssertNotNil(CameraError.deviceUnavailable.errorDescription)
    }

    func test_outputUnavailable_hasDescription() {
        XCTAssertNotNil(CameraError.outputUnavailable.errorDescription)
    }

    // MARK: - Initialisation

    func test_init_createsPreviewLayer() {
        let session = CameraSession()
        XCTAssertNotNil(session.previewLayer)
    }

    func test_framePublisher_emitsNothingBeforeStart() {
        let session = CameraSession()
        var received = false
        let cancellable = session.framePublisher.sink { _ in received = true }
        XCTAssertFalse(received)
        _ = cancellable
    }

    // MARK: - start() permission guard

    func test_start_throwsCameraError_whenNotAuthorized() {
        // In the test environment AVCaptureDevice is not authorized (.notDetermined or .denied),
        // so start() must throw CameraError.permissionDenied.
        let session = CameraSession()
        XCTAssertThrowsError(try session.start()) { error in
            XCTAssertTrue(error is CameraError, "Expected CameraError, got \(error)")
            if case CameraError.permissionDenied = error { } else {
                XCTFail("Expected .permissionDenied, got \(error)")
            }
        }
    }

    // MARK: - stop()

    func test_stop_doesNotCrashWhenNeverStarted() {
        let session = CameraSession()
        session.stop()   // Must not throw or crash
    }

    // MARK: - Protocol conformance

    func test_conformsToCameraSessionProtocol() {
        let session: CameraSessionProtocol = CameraSession()
        XCTAssertNotNil(session.previewLayer)
    }
}
