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
}
