import XCTest
import CoreML
@testable import ShopliftDetect

final class PoseNormalizerTests: XCTestCase {

    func testResolutionDivisionByWidthAndHeight() {
        XCTFail("Not implemented")
    }

    func testMeanSubtractionOverAllFramesAndJoints() {
        XCTFail("Not implemented")
    }

    func testYAxisStdAppliedToBothXandY() {
        XCTFail("Not implemented")
    }

    func testConfidencePassesThroughUnmodified() {
        XCTFail("Not implemented")
    }

    func testMatchesPythonOutputForSeededFixture() throws {
        // Loads ios/ShopliftDetectTests/Fixtures/normal_pose_window.json
        // Compares to expected_output with tolerance 1e-5.
        XCTFail("Not implemented")
    }

    func testZeroVarianceDoesNotProduceNaN() {
        XCTFail("Not implemented")
    }

    func testOutputShapeIs_1_2_24_18() throws {
        XCTFail("Not implemented")
    }

    func testUsesPixelCoordinatesNotNormalized() {
        XCTFail("Not implemented")
    }
}
