import Foundation

enum AnomalyLabel: Sendable, Equatable {
    case normal
    case anomaly
    case warmup
}

struct AnomalyResult: Sendable, Equatable {
    let score: Float
    let label: AnomalyLabel
    let timestamp: Date
}
