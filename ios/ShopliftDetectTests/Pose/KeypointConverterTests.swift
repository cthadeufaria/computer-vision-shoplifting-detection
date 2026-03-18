import XCTest
import CoreMedia
@testable import ShopliftDetect

final class KeypointConverterTests: XCTestCase {

    // MARK: - Fixture loading

    private func loadFixtureData(_ name: String) throws -> Data {
        if let url = Bundle(for: Self.self).url(forResource: name, withExtension: "json") {
            return try Data(contentsOf: url)
        }
        // Fallback: source-relative path for local development.
        let sourceFile = URL(fileURLWithPath: #filePath)
        let fixturesDir = sourceFile
            .deletingLastPathComponent()  // .../Pose/
            .deletingLastPathComponent()  // .../ShopliftDetectTests/
            .appendingPathComponent("Fixtures")
        return try Data(contentsOf: fixturesDir.appendingPathComponent("\(name).json"))
    }

    private struct CocoFixture {
        let input17: [Keypoint]
        let output18: [Keypoint]
        let expectedNeckXY: (Float, Float)
        let expectedNeckConf: Float
        let checks: [String: Int]
    }

    private func loadCocoFixture() throws -> CocoFixture {
        let data = try loadFixtureData("coco17_sample")
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let raw17 = json["input_coco17"] as! [[Double]]
        let raw18 = json["output_coco18"] as! [[Double]]
        let neckXY = json["expected_neck_xy"] as! [Double]
        let neckConf = json["expected_neck_conf"] as! Double
        let checks = json["checks"] as! [String: Int]
        return CocoFixture(
            input17: raw17.map { Keypoint(x: Float($0[0]), y: Float($0[1]), confidence: Float($0[2])) },
            output18: raw18.map { Keypoint(x: Float($0[0]), y: Float($0[1]), confidence: Float($0[2])) },
            expectedNeckXY: (Float(neckXY[0]), Float(neckXY[1])),
            expectedNeckConf: Float(neckConf),
            checks: checks
        )
    }

    // MARK: - Tests

    func testOutputHas18Keypoints() throws {
        let fixture = try loadCocoFixture()
        let neck = KeypointConverter.selectNeck(
            visionNeck: nil,
            leftShoulder: fixture.input17[5],
            rightShoulder: fixture.input17[6]
        )
        let result = KeypointConverter.reorder(coco17: fixture.input17, neck: neck)
        XCTAssertEqual(result.count, 18)
    }

    func testNeckIsSyntheticAverageOfShoulders() {
        let ls = Keypoint(x: 100, y: 200, confidence: 0.8)
        let rs = Keypoint(x: 200, y: 200, confidence: 0.6)
        let neck = KeypointConverter.selectNeck(visionNeck: nil, leftShoulder: ls, rightShoulder: rs)
        XCTAssertEqual(neck.x, 150, accuracy: 1e-5)
        XCTAssertEqual(neck.y, 200, accuracy: 1e-5)
        XCTAssertEqual(neck.confidence, 0.7, accuracy: 1e-5)
    }

    func testOpenPoseReorderingMatchesPythonOppOrder() throws {
        let fixture = try loadCocoFixture()
        let neck = KeypointConverter.selectNeck(
            visionNeck: nil,
            leftShoulder: fixture.input17[5],
            rightShoulder: fixture.input17[6]
        )
        let result = KeypointConverter.reorder(coco17: fixture.input17, neck: neck)
        var maxDiff: Float = 0
        for i in 0..<18 {
            maxDiff = max(maxDiff, abs(result[i].x - fixture.output18[i].x))
            maxDiff = max(maxDiff, abs(result[i].y - fixture.output18[i].y))
            maxDiff = max(maxDiff, abs(result[i].confidence - fixture.output18[i].confidence))
        }
        XCTAssertLessThan(maxDiff, 1e-4, "Reordered keypoints don't match Python fixture (max diff \(maxDiff))")
    }

    func testNoseRemainsAtIndex0() throws {
        let fixture = try loadCocoFixture()
        let neck = KeypointConverter.selectNeck(
            visionNeck: nil,
            leftShoulder: fixture.input17[5],
            rightShoulder: fixture.input17[6]
        )
        let result = KeypointConverter.reorder(coco17: fixture.input17, neck: neck)
        let noseIdx = fixture.checks["nose_index_in_output"]!
        XCTAssertEqual(noseIdx, 0)
        XCTAssertEqual(result[noseIdx].x, fixture.input17[0].x, accuracy: 1e-5)
        XCTAssertEqual(result[noseIdx].y, fixture.input17[0].y, accuracy: 1e-5)
    }

    func testRightShoulderLandsAtOpenPoseIndex2() throws {
        let fixture = try loadCocoFixture()
        let neck = KeypointConverter.selectNeck(
            visionNeck: nil,
            leftShoulder: fixture.input17[5],
            rightShoulder: fixture.input17[6]
        )
        let result = KeypointConverter.reorder(coco17: fixture.input17, neck: neck)
        let idx = fixture.checks["right_shoulder_index_in_output"]!
        XCTAssertEqual(idx, 2)
        XCTAssertEqual(result[idx].x, fixture.input17[6].x, accuracy: 1e-5)  // coco17[6] = rightShoulder
        XCTAssertEqual(result[idx].y, fixture.input17[6].y, accuracy: 1e-5)
    }

    func testLeftShoulderLandsAtOpenPoseIndex5() throws {
        let fixture = try loadCocoFixture()
        let neck = KeypointConverter.selectNeck(
            visionNeck: nil,
            leftShoulder: fixture.input17[5],
            rightShoulder: fixture.input17[6]
        )
        let result = KeypointConverter.reorder(coco17: fixture.input17, neck: neck)
        let idx = fixture.checks["left_shoulder_index_in_output"]!
        XCTAssertEqual(idx, 5)
        XCTAssertEqual(result[idx].x, fixture.input17[5].x, accuracy: 1e-5)  // coco17[5] = leftShoulder
        XCTAssertEqual(result[idx].y, fixture.input17[5].y, accuracy: 1e-5)
    }

    func testUsesVisionNeckWhenHighConfidence() {
        let visionNeck = Keypoint(x: 300, y: 400, confidence: 0.95)
        let ls = Keypoint(x: 100, y: 200, confidence: 0.8)
        let rs = Keypoint(x: 200, y: 200, confidence: 0.6)
        let neck = KeypointConverter.selectNeck(visionNeck: visionNeck, leftShoulder: ls, rightShoulder: rs)
        XCTAssertEqual(neck.x, 300, accuracy: 1e-5)
        XCTAssertEqual(neck.y, 400, accuracy: 1e-5)
        XCTAssertEqual(neck.confidence, 0.95, accuracy: 1e-5)
    }

    func testFallsBackToSyntheticNeckWhenLowConfidence() {
        // Confidence < 0.3 → ignore Vision neck, use shoulder average.
        let lowConfNeck = Keypoint(x: 999, y: 999, confidence: 0.1)
        let ls = Keypoint(x: 100, y: 200, confidence: 0.8)
        let rs = Keypoint(x: 200, y: 200, confidence: 0.6)
        let neck = KeypointConverter.selectNeck(visionNeck: lowConfNeck, leftShoulder: ls, rightShoulder: rs)
        XCTAssertEqual(neck.x, 150, accuracy: 1e-5)
        XCTAssertEqual(neck.y, 200, accuracy: 1e-5)
    }

    func testConfidenceValuesPreservedThroughConversion() throws {
        let fixture = try loadCocoFixture()
        let neck = KeypointConverter.selectNeck(
            visionNeck: nil,
            leftShoulder: fixture.input17[5],
            rightShoulder: fixture.input17[6]
        )
        let result = KeypointConverter.reorder(coco17: fixture.input17, neck: neck)
        // opp_order[0]=0 → nose at index 0
        XCTAssertEqual(result[0].confidence, fixture.input17[0].confidence, accuracy: 1e-5)
        // opp_order[2]=6 → rightShoulder at index 2
        XCTAssertEqual(result[2].confidence, fixture.input17[6].confidence, accuracy: 1e-5)
        // opp_order[5]=5 → leftShoulder at index 5
        XCTAssertEqual(result[5].confidence, fixture.input17[5].confidence, accuracy: 1e-5)
    }

    func testZeroInputProducesZeroNeck() {
        let zeros = [Keypoint](repeating: Keypoint(x: 0, y: 0, confidence: 0), count: 17)
        let neck = KeypointConverter.selectNeck(
            visionNeck: nil,
            leftShoulder: zeros[5],
            rightShoulder: zeros[6]
        )
        XCTAssertEqual(neck.x, 0, accuracy: 1e-5)
        XCTAssertEqual(neck.y, 0, accuracy: 1e-5)
        XCTAssertEqual(neck.confidence, 0, accuracy: 1e-5)
    }
}
