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

    override func setUp() {
        mockCamera = MockCameraSession()
        mockEstimator = MockPoseEstimator()
        mockConverter = MockKeypointConverter()
        mockScorer = MockAnomalyScorer()
        mockTracking = MockTrackingService()
        sut = DetectionViewModel(
            camera: mockCamera,
            estimator: mockEstimator,
            converter: mockConverter,
            scorer: mockScorer,
            tracking: mockTracking
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
}
