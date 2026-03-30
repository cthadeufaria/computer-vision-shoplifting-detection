protocol AnomalyScorerProtocol: Sendable {
    func classify(score: Float, isWarmup: Bool) -> AnomalyResult
}
