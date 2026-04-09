import CoreGraphics
import Foundation

struct DetectionResult: Equatable, Sendable {
    let trackID: Int
    let score: Float
    let label: AnomalyLabel
    let keypoints: [Keypoint]
    let boundingBox: CGRect
    let timestamp: Date
}
