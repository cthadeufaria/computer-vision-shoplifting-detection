import XCTest
@testable import ShopliftDetect

final class KeypointConverterTests: XCTestCase {

    private let previewSize = CGSize(width: 640, height: 480)

    // MARK: - Tests

    func testOutputHas18Keypoints() throws {
        XCTFail("Not implemented")
    }

    func testNeckIsSyntheticAverageOfShoulders() throws {
        XCTFail("Not implemented")
    }

    func testOpenPoseReorderingMatchesPythonOppOrder() throws {
        XCTFail("Not implemented")
    }

    func testNoseRemainsAtIndex0() throws {
        XCTFail("Not implemented")
    }

    func testRightShoulderLandsAtOpenPoseIndex2() throws {
        XCTFail("Not implemented")
    }

    func testLeftShoulderLandsAtOpenPoseIndex5() throws {
        XCTFail("Not implemented")
    }

    func testUsesVisionNeckWhenHighConfidence() throws {
        XCTFail("Not implemented")
    }

    func testFallsBackToSyntheticNeckWhenLowConfidence() throws {
        XCTFail("Not implemented")
    }

    func testConfidenceValuesPreservedThroughConversion() throws {
        XCTFail("Not implemented")
    }

    func testZeroInputProducesZeroNeck() throws {
        XCTFail("Not implemented")
    }
}
