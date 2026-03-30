import CoreGraphics
import Foundation

@MainActor
protocol TrackingServiceProtocol: AnyObject {
    func matchTrack(for skeleton: PoseSkeleton) -> String
}

/// Assigns persistent track IDs to detected skeletons using IoU-based matching.
@MainActor
final class TrackingService: TrackingServiceProtocol {
    private let iouThreshold: CGFloat = 0.3
    private var lastBBox: [String: CGRect] = [:]

    func matchTrack(for skeleton: PoseSkeleton) -> String {
        var bestID: String?
        var bestIoU: CGFloat = 0

        for (id, bbox) in lastBBox {
            let iou = computeIoU(skeleton.boundingBox, bbox)
            if iou > bestIoU {
                bestIoU = iou
                bestID = id
            }
        }

        if let id = bestID, bestIoU >= iouThreshold {
            lastBBox[id] = skeleton.boundingBox
            return id
        }

        let newID = UUID().uuidString
        lastBBox[newID] = skeleton.boundingBox
        return newID
    }

    private func computeIoU(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let intersection = a.intersection(b)
        guard !intersection.isNull else { return 0 }
        let intersectionArea = intersection.width * intersection.height
        let unionArea = a.width * a.height + b.width * b.height - intersectionArea
        return unionArea > 0 ? intersectionArea / unionArea : 0
    }
}
