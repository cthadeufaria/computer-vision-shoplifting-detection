import AVFoundation
import Combine
@testable import ShopliftDetect

final class MockCameraSession: CameraSessionProtocol {
    var startCallCount = 0
    var stopCallCount = 0
    var shouldThrowOnStart = false

    let previewLayer = AVCaptureVideoPreviewLayer()
    private let subject = PassthroughSubject<CVPixelBuffer, Never>()
    var framePublisher: AnyPublisher<CVPixelBuffer, Never> { subject.eraseToAnyPublisher() }

    func start() throws {
        if shouldThrowOnStart { throw CameraError.permissionDenied }
        startCallCount += 1
    }

    func stop() {
        stopCallCount += 1
    }

    /// Push a frame through the publisher for testing processFrame paths.
    func emit(_ pixelBuffer: CVPixelBuffer) {
        subject.send(pixelBuffer)
    }
}
