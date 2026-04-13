import CoreVideo
import XCTest
@testable import ShopliftDetect

@MainActor
final class PosePreviewViewModelTests: XCTestCase {
    var sut: PosePreviewViewModel!
    var mockCamera: MockCameraSession!
    var mockEstimator: MockPoseEstimator!
    var mockConverter: MockKeypointConverter!

    override func setUp() {
        mockCamera = MockCameraSession()
        mockEstimator = MockPoseEstimator()
        mockConverter = MockKeypointConverter()
        sut = PosePreviewViewModel(
            camera: mockCamera,
            estimator: mockEstimator,
            converter: mockConverter
        )
    }

    func test_start_callsCameraStart() throws {
        try sut.start()
        XCTAssertEqual(mockCamera.startCallCount, 1)
    }

    func test_start_whenCameraThrows_propagatesError() {
        mockCamera.shouldThrowOnStart = true
        XCTAssertThrowsError(try sut.start())
    }

    func test_stop_callsCameraStop() throws {
        try sut.start()
        sut.stop()
        XCTAssertEqual(mockCamera.stopCallCount, 1)
    }

    func test_stop_clearsSkeletons() throws {
        try sut.start()
        sut.stop()
        XCTAssertTrue(sut.skeletons.isEmpty)
    }

    func test_enablePreviewTestMode_setsDebugSkeletonState() {
        sut.enablePreviewTestMode()

        XCTAssertEqual(sut.skeletons.count, 1)
        XCTAssertEqual(sut.debugInfo, "UI Preview Mode")
    }

    func test_start_dropsNewFramesWhilePreviousFrameIsStillProcessing() throws {
        mockEstimator.detectionDelayNanoseconds = 200_000_000

        try sut.start()
        mockCamera.emit(makePixelBuffer())
        mockCamera.emit(makePixelBuffer())

        let processed = expectation(description: "frame processed")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
            processed.fulfill()
        }
        wait(for: [processed], timeout: 1.0)

        XCTAssertEqual(mockEstimator.detectCallCount, 1)
        XCTAssertEqual(mockEstimator.maxConcurrentCalls, 1)
    }

    func test_stopWhileFrameIsProcessing_discardsLatePreviewUpdates() throws {
        mockEstimator.detectionDelayNanoseconds = 200_000_000

        try sut.start()
        mockCamera.emit(makePixelBuffer())
        sut.stop()

        let processed = expectation(description: "late frame ignored")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
            processed.fulfill()
        }
        wait(for: [processed], timeout: 1.0)

        XCTAssertEqual(mockCamera.stopCallCount, 1)
        XCTAssertTrue(sut.skeletons.isEmpty)
        XCTAssertEqual(sut.debugInfo, "")
    }

    func test_stopAndRestart_allowsFrameProcessingAgain() throws {
        try sut.start()
        sut.stop()

        try sut.start()
        mockCamera.emit(makePixelBuffer())

        let processed = expectation(description: "restart frame processed")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            processed.fulfill()
        }
        wait(for: [processed], timeout: 1.0)

        XCTAssertEqual(mockCamera.startCallCount, 2)
        XCTAssertEqual(mockEstimator.detectCallCount, 1)
    }

    private func makePixelBuffer() -> CVPixelBuffer {
        var pixelBuffer: CVPixelBuffer?
        CVPixelBufferCreate(
            kCFAllocatorDefault,
            4,
            4,
            kCVPixelFormatType_32BGRA,
            nil,
            &pixelBuffer
        )
        return try! XCTUnwrap(pixelBuffer)
    }
}
