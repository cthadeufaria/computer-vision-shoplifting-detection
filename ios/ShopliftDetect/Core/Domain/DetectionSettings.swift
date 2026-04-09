import Foundation

struct DetectionSettings: Sendable, Equatable {
    var anomalyThreshold: Float

    static let `default` = DetectionSettings(anomalyThreshold: -1.2)
}
