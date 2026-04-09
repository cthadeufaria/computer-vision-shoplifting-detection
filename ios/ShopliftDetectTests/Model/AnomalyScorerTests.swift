import XCTest
@testable import ShopliftDetect

final class AnomalyScorerTests: XCTestCase {

    private var scorer = AnomalyScorer()

    func testScoreBelowThresholdIsAnomaly() {
        let result = scorer.classify(score: -2.0, isWarmup: false)
        XCTAssertEqual(result.label, .anomaly)
    }

    func testScoreAboveThresholdIsNormal() {
        let result = scorer.classify(score: 0.0, isWarmup: false)
        XCTAssertEqual(result.label, .normal)
    }

    func testScoreAtThresholdIsAnomaly() {
        // Boundary: score == threshold → ANOMALY (uses <=)
        let result = scorer.classify(score: -1.2, isWarmup: false)
        XCTAssertEqual(result.label, .anomaly)
    }

    func testDefaultThresholdIsNegative1Point2() {
        XCTAssertEqual(scorer.threshold, -1.2, accuracy: 1e-6)
    }

    func testThresholdIsSettable() {
        scorer.threshold = -0.5
        let result = scorer.classify(score: -0.8, isWarmup: false)
        XCTAssertEqual(result.label, .anomaly)
    }

    func testResultContainsScoreAndTimestamp() {
        let before = Date()
        let result = scorer.classify(score: -1.5, isWarmup: false)
        let after = Date()
        XCTAssertEqual(result.score, -1.5, accuracy: 1e-6)
        XCTAssertGreaterThanOrEqual(result.timestamp, before)
        XCTAssertLessThanOrEqual(result.timestamp, after)
    }

    func testWarmupFlagTrueWhenBufferNotFull() {
        let result = scorer.classify(score: -2.0, isWarmup: true)
        XCTAssertEqual(result.label, .warmup)
    }

    func testWarmupFlagFalseAfterFirstFullWindow() {
        let result = scorer.classify(score: -2.0, isWarmup: false)
        XCTAssertNotEqual(result.label, .warmup)
    }

    func test_init_customThreshold_treatsScoreBelowAsAnomaly() {
        let sut = AnomalyScorer(threshold: -0.5)
        XCTAssertEqual(sut.classify(score: -0.6, isWarmup: false).label, .anomaly)
        XCTAssertEqual(sut.classify(score: 0.0, isWarmup: false).label, .normal)
    }

    func test_init_defaultThresholdIsNegative1Point2() {
        let sut = AnomalyScorer()
        XCTAssertEqual(sut.threshold, -1.2, accuracy: 1e-6)
    }

    func test_detectionSettingsDefaultUsesNegative1Point2Threshold() {
        XCTAssertEqual(DetectionSettings.default.anomalyThreshold, -1.2, accuracy: 1e-6)
    }

    @MainActor
    func test_settingsServicePersistsThresholdThroughPersistenceService() {
        let persistence = MockPersistenceService()
        let sut = UserDefaultsSettingsService(persistence: persistence)

        sut.anomalyThreshold = -0.8

        XCTAssertEqual(persistence.detectionSettings.anomalyThreshold, -0.8, accuracy: 1e-6)
        XCTAssertEqual(sut.anomalyThreshold, -0.8, accuracy: 1e-6)
    }
}
