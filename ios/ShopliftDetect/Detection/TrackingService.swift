import CoreGraphics
import Foundation

@MainActor
protocol TrackingServiceProtocol: AnyObject {
    func matchTrack(for skeleton: PoseSkeleton) -> String
}

/// Assigns persistent track IDs to detected skeletons using IoU-based matching.
@MainActor
final class TrackingService: TrackingServiceProtocol {
    private struct TrackState {
        var boundingBox: CGRect
        var lastSeenFrame: Int
    }

    private let iouThreshold: CGFloat = 0.3
    private let maxMissingFrames = 6
    private var tracks: [String: TrackState] = [:]
    private var currentFrameIndex: Int?
    private var claimedTrackIDs = Set<String>()

    func matchTrack(for skeleton: PoseSkeleton) -> String {
        if currentFrameIndex != skeleton.frameIndex {
            currentFrameIndex = skeleton.frameIndex
            claimedTrackIDs.removeAll()
        }

        pruneExpiredTracks(currentFrameIndex: skeleton.frameIndex)

        var bestID: String?
        var bestIoU: CGFloat = 0

        for (id, state) in tracks where !claimedTrackIDs.contains(id) {
            let iou = computeIoU(skeleton.boundingBox, state.boundingBox)
            if iou > bestIoU {
                bestIoU = iou
                bestID = id
            }
        }

        if let id = bestID, bestIoU >= iouThreshold {
            tracks[id] = TrackState(
                boundingBox: skeleton.boundingBox,
                lastSeenFrame: skeleton.frameIndex
            )
            claimedTrackIDs.insert(id)
            return id
        }

        let newID = UUID().uuidString
        tracks[newID] = TrackState(
            boundingBox: skeleton.boundingBox,
            lastSeenFrame: skeleton.frameIndex
        )
        claimedTrackIDs.insert(newID)
        return newID
    }

    private func pruneExpiredTracks(currentFrameIndex: Int) {
        tracks = tracks.filter { _, state in
            currentFrameIndex - state.lastSeenFrame <= maxMissingFrames
        }
    }

    private func computeIoU(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let intersection = a.intersection(b)
        guard !intersection.isNull else { return 0 }
        let intersectionArea = intersection.width * intersection.height
        let unionArea = a.width * a.height + b.width * b.height - intersectionArea
        return unionArea > 0 ? intersectionArea / unionArea : 0
    }
}
