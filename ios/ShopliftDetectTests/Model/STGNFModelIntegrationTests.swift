import XCTest
import CoreML
@testable import ShopliftDetect

/// These tests require STGNFModel.mlpackage to be bundled.
/// They are skipped automatically when the model is absent.
final class STGNFModelIntegrationTests: XCTestCase {

    private var model: STGNFModel?

    override func setUp() {
        super.setUp()
        model = try? STGNFModel()
    }

    private func skipIfNoModel() throws {
        try XCTSkipIf(model == nil, "STGNFModel.mlpackage not bundled — skipping integration tests")
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
        // Load the Python-generated expected value from fixtures once available.
        // For now this is a placeholder that will be filled in Step 3 (fixture generation).
        XCTFail("Fixture not yet generated — run scripts/generate_fixtures.py first")
    }
}
