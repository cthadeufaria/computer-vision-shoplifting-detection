import Foundation

struct AnomalyScorer: Sendable, AnomalyScorerProtocol {
    var threshold: Float

    init(threshold: Float = -1.2) {
        self.threshold = threshold
    }

    /// - Parameters:
    ///   - score: anomaly_score = -NLL from STG-NF (more negative = more anomalous).
    ///   - isWarmup: true when the 24-frame buffer is not yet filled.
    func classify(score: Float, isWarmup: Bool) -> AnomalyResult {
        let label: AnomalyLabel = isWarmup ? .warmup : (score <= threshold ? .anomaly : .normal)
        return AnomalyResult(score: score, label: label, timestamp: Date())
    }
}
