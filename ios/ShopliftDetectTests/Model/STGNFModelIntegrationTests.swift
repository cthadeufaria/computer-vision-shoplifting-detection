import XCTest
import CoreML
@testable import ShopliftDetect

/// These tests require the raw STGNF model asset to be bundled.
/// They are skipped automatically when the model is absent.
final class STGNFModelIntegrationTests: XCTestCase {

    private var model: STGNFModelRunner?

    override func setUp() {
        super.setUp()
        model = try? STGNFModelRunner()
    }

    private func skipIfNoModel() throws {
        try XCTSkipIf(model == nil, "STGNFModel model asset not bundled — skipping integration tests")
    }

    private func loadFixtureData(_ name: String) throws -> Data {
        if let url = Bundle(for: Self.self).url(forResource: name, withExtension: "json") {
            return try Data(contentsOf: url)
        }
        let sourceFile = URL(fileURLWithPath: #filePath)
        let fixturesDir = sourceFile
            .deletingLastPathComponent()  // .../Model/
            .deletingLastPathComponent()  // .../ShopliftDetectTests/
            .appendingPathComponent("Fixtures")
        return try Data(contentsOf: fixturesDir.appendingPathComponent("\(name).json"))
    }

    func testModelLoadsFromBundle() throws {
        try skipIfNoModel()
        XCTAssertNotNil(model)
    }

    func testModelOutputIsFiniteOnValidInput() throws {
        try skipIfNoModel()
        // Zero-filled input is intentionally excluded: a normalizing flow assigns -inf
        // log-likelihood to degenerate (zero-variance) inputs, which is correct behaviour.
        // Use the inference_nll_sample fixture — a pre-normalised non-degenerate tensor
        // whose finite output is independently verified in testCoreMLMatchesPythonNLLWithin1e3.
        let data = try loadFixtureData("inference_nll_sample")
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let inputRaw = json["input_pose_window"] as! [[[[Double]]]]

        let array = try MLMultiArray(shape: [1, 2, 24, 18], dataType: .float32)
        for ch in 0..<2 {
            for f in 0..<24 {
                for j in 0..<18 {
                    array[ch * 24 * 18 + f * 18 + j] = NSNumber(value: Float(inputRaw[0][ch][f][j]))
                }
            }
        }

        let score = try XCTUnwrap(model).runInference(on: array)
        XCTAssertTrue(score.isFinite, "Expected finite output for valid non-degenerate input, got \(score)")
    }

    func testNormalPoseFixtureProducesFiniteOutput() throws {
        try skipIfNoModel()
        // Load the pre-normalised expected_output from normal_pose_window.json
        // (shape [1][24][18][3] = [batch][frame][joint][x, y, conf]).
        // Confidence is dropped; x → channel 0, y → channel 1 of [1,2,24,18].
        let data = try loadFixtureData("normal_pose_window")
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let expected = json["expected_output"] as! [[[[Double]]]]

        let array = try MLMultiArray(shape: [1, 2, 24, 18], dataType: .float32)
        for f in 0..<24 {
            for j in 0..<18 {
                array[0 * 24 * 18 + f * 18 + j] = NSNumber(value: Float(expected[0][f][j][0]))  // x
                array[1 * 24 * 18 + f * 18 + j] = NSNumber(value: Float(expected[0][f][j][1]))  // y
            }
        }

        let score = try XCTUnwrap(model).runInference(on: array)
        XCTAssertTrue(score.isFinite, "Expected finite score for realistic pose fixture, got \(score)")
    }

    func testSingleWindowInferenceUnder50ms() throws {
        try skipIfNoModel()
        let input = try MLMultiArray(shape: [1, 2, 24, 18], dataType: .float32)
        let start = Date()
        _ = try XCTUnwrap(model).runInference(on: input)
        let elapsed = Date().timeIntervalSince(start) * 1000
        XCTAssertLessThan(elapsed, 50, "Inference took \(elapsed)ms, expected <50ms")
    }

    func testCoreMLMatchesPythonNLLWithin1e3() throws {
        try skipIfNoModel()
        let data = try loadFixtureData("inference_nll_sample")
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let inputRaw = json["input_pose_window"] as! [[[[Double]]]]  // [1][2][24][18]
        let expectedNLL = Float(json["expected_nll"] as! Double)

        // Build MLMultiArray [1, 2, 24, 18] from fixture.
        let array = try MLMultiArray(shape: [1, 2, 24, 18], dataType: .float32)
        for ch in 0..<2 {
            for f in 0..<24 {
                for j in 0..<18 {
                    array[ch * 24 * 18 + f * 18 + j] = NSNumber(value: Float(inputRaw[0][ch][f][j]))
                }
            }
        }

        // runInference returns anomaly_score = -NLL, so NLL = -score.
        let score = try XCTUnwrap(model).runInference(on: array)
        let coremlNLL = -score
        XCTAssertEqual(coremlNLL, expectedNLL, accuracy: 1e-3,
                       "CoreML NLL \(coremlNLL) differs from PyTorch NLL \(expectedNLL)")
    }

    func testModelContractMatchesApr01_1416RunAssumptions() {
        XCTAssertEqual(STGNFModelRunner.expectedInputShape, [1, 2, 24, 18])
        XCTAssertEqual(STGNFModelRunner.expectedSegmentLength, 24)
        XCTAssertEqual(STGNFModelRunner.expectedJointCount, 18)
        XCTAssertFalse(STGNFModelRunner.usesConfidenceChannel)
        XCTAssertEqual(STGNFModelRunner.outputFeatureName, "nll_score")
    }
}
