import Foundation
@testable import ShopliftDetect

final class MockAnomalyScorer: AnomalyScorerProtocol, @unchecked Sendable {
    var threshold: Float = DetectionSettings.default.anomalyThreshold
    var stubbedResult = AnomalyResult(score: 0.0, label: .normal, timestamp: Date())
    var classifyCallCount = 0

    func classify(score: Float, isWarmup: Bool) -> AnomalyResult {
        classifyCallCount += 1
        return stubbedResult
    }
}
