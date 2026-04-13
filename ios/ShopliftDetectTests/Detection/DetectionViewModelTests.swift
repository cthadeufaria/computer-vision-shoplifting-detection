import CoreVideo
import XCTest
@testable import ShopliftDetect

@MainActor
final class DetectionViewModelTests: XCTestCase {
    var sut: DetectionViewModel!
    var mockCamera: MockCameraSession!
    var mockEstimator: MockPoseEstimator!
    var mockConverter: MockKeypointConverter!
    var mockScorer: MockAnomalyScorer!
    var mockTracking: MockTrackingService!
    var mockSettings: MockSettingsService!
    var mockStreaming: MockStreamingService!

    override func setUp() {
        mockCamera = MockCameraSession()
        mockEstimator = MockPoseEstimator()
        mockConverter = MockKeypointConverter()
        mockScorer = MockAnomalyScorer()
        mockTracking = MockTrackingService()
        mockSettings = MockSettingsService()
        mockStreaming = MockStreamingService()
        sut = DetectionViewModel(
            camera: mockCamera,
            estimator: mockEstimator,
            converter: mockConverter,
            scorer: mockScorer,
            tracking: mockTracking,
            settings: mockSettings,
            streaming: mockStreaming
        )
    }

    // MARK: - start()

    func test_start_callsCameraStart() throws {
        try sut.start()
        XCTAssertEqual(mockCamera.startCallCount, 1)
    }

    func test_start_setsStateToWarmingUp() throws {
        try sut.start()
        if case .warmingUp = sut.detectionState { /* pass */ } else {
            XCTFail("Expected .warmingUp, got \(sut.detectionState)")
        }
    }

    func test_start_startsStreaming() throws {
        try sut.start()
        XCTAssertEqual(mockStreaming.startCallCount, 1)
        XCTAssertTrue(sut.isStreaming)
    }

    func test_start_whenCameraThrows_stateRemainsIdle() {
        mockCamera.shouldThrowOnStart = true
        XCTAssertThrowsError(try sut.start())
        XCTAssertEqual(sut.detectionState, .idle)
    }

    // MARK: - stop()

    func test_stop_setsStateToIdle() throws {
        try sut.start()
        sut.stop()
        XCTAssertEqual(sut.detectionState, .idle)
    }

    func test_stop_callsCameraStop() throws {
        try sut.start()
        sut.stop()
        XCTAssertEqual(mockCamera.stopCallCount, 1)
    }

    func test_stop_stopsStreaming() throws {
        try sut.start()
        sut.stop()
        XCTAssertEqual(mockStreaming.stopCallCount, 1)
        XCTAssertFalse(sut.isStreaming)
    }

    func test_stop_clearsSkeletons() throws {
        try sut.start()
        sut.stop()
        XCTAssertTrue(sut.skeletons.isEmpty)
    }

    // MARK: - enablePreviewTestMode()

    func test_enablePreviewTestMode_setsWarmingUpState() {
        sut.enablePreviewTestMode()
        if case .warmingUp = sut.detectionState { /* pass */ } else {
            XCTFail("Expected .warmingUp")
        }
    }

    func test_threshold_readsFromSettings() {
        mockSettings.anomalyThreshold = -0.7
        XCTAssertEqual(sut.threshold, -0.7, accuracy: 0.001)
    }

    func test_updateThreshold_updatesSettingsAndScorer() {
        sut.updateThreshold(-0.4)
        XCTAssertEqual(mockSettings.anomalyThreshold, -0.4, accuracy: 0.001)
        XCTAssertEqual(mockScorer.threshold, -0.4, accuracy: 0.001)
        XCTAssertEqual(sut.threshold, -0.4, accuracy: 0.001)
    }

    func test_stopLeavesLastPublishedFrameAvailableForSupervisorTile() throws {
        try sut.start()
        mockStreaming.feedStates = [
            SupervisorFeedTileState(
                sessionID: UUID(),
                deviceName: "Aisle 3 Camera",
                connectionState: .stale,
                latestFrame: VideoFrame(timestamp: 1, jpegData: Data([0x01]), width: 32, height: 32),
                latestDetections: []
            )
        ]

        sut.stop()

        XCTAssertNotNil(mockStreaming.feedStates.first?.latestFrame)
        XCTAssertEqual(mockStreaming.stopCallCount, 1)
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

    func test_stopWhileFrameIsProcessing_keepsSupervisorFeedStale() throws {
        mockEstimator.detectionDelayNanoseconds = 200_000_000
        let sessionID = UUID()
        mockStreaming.feedStates = [
            SupervisorFeedTileState(
                sessionID: sessionID,
                deviceName: "Aisle 3 Camera",
                connectionState: .connected,
                latestFrame: nil,
                latestDetections: []
            )
        ]

        try sut.start()
        mockCamera.emit(makePixelBuffer())
        sut.stop()

        let processed = expectation(description: "late frame ignored")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
            processed.fulfill()
        }
        wait(for: [processed], timeout: 1.0)

        XCTAssertEqual(mockStreaming.feedStates.first?.connectionState, .stale)
        XCTAssertFalse(mockStreaming.isStreaming)
    }

    func test_stopAndRestart_reentersWarmupAndRestartsStreaming() throws {
        try sut.start()
        sut.stop()
        try sut.start()

        if case .warmingUp = sut.detectionState {
            // pass
        } else {
            XCTFail("Expected warmup after restart")
        }
        XCTAssertEqual(mockCamera.startCallCount, 2)
        XCTAssertEqual(mockStreaming.startCallCount, 2)
        XCTAssertTrue(sut.isStreaming)
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
