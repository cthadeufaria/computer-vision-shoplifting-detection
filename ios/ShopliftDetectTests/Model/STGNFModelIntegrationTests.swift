import XCTest
import CoreML
@testable import ShopliftDetect

/// These tests require STGNFModel.mlpackage to be bundled.
/// They are skipped automatically when the model is absent.
final class STGNFModelIntegrationTests: XCTestCase {

    private var model: STGNFModelRunner?

    override func setUp() {
        super.setUp()
        model = try? STGNFModelRunner()
    }

    private func skipIfNoModel() throws {
        try XCTSkipIf(model == nil, "STGNFModel.mlpackage not bundled — skipping integration tests")
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

    func testModelOutputIsFiniteOnZeroInput() throws {
        try skipIfNoModel()
        let input = try MLMultiArray(shape: [1, 2, 24, 18], dataType: .float32)
        let score = try XCTUnwrap(model).runInference(on: input)
        XCTAssertTrue(score.isFinite, "Expected finite output for zero input")
    }

    func testNormalPoseFixtureProducesHighScore() throws {
        try skipIfNoModel()
        // A normal (non-anomalous) pose window should produce score > threshold.
        let input = try MLMultiArray(shape: [1, 2, 24, 18], dataType: .float32)
        let score = try XCTUnwrap(model).runInference(on: input)
        XCTAssertGreaterThan(score, AnomalyScorer().threshold)
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
}
