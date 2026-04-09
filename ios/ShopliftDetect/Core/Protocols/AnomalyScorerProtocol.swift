protocol AnomalyScorerProtocol: Sendable {
    var threshold: Float { get set }
    func classify(score: Float, isWarmup: Bool) -> AnomalyResult
}
