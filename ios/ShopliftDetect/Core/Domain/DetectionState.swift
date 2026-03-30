import Foundation

enum DetectionState: Sendable, Equatable {
    case idle
    case warmingUp(framesCollected: Int, framesNeeded: Int)
    case running(latestResult: AnomalyResult)
    case error(reason: String)
}
