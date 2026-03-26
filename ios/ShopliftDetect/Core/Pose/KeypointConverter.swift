import Vision

/// Converts VNHumanBodyPoseObservation (17 COCO keypoints) to COCO18 format
/// using OpenPose reindexing (opp_order), matching Python's keypoints17_to_coco18().
struct KeypointConverter {
    // Mirrors Python: opp_order = [0,17,6,8,10,5,7,9,12,14,16,11,13,15,2,1,4,3]
    static let oppOrder: [Int] = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]

    /// VNHumanBodyPoseObservation joint names in COCO17 order (index 0..16).
    static let coco17Joints: [VNHumanBodyPoseObservation.JointName] = [
        .nose,
        .leftEye, .rightEye,
        .leftEar, .rightEar,
        .leftShoulder, .rightShoulder,
        .leftElbow, .rightElbow,
        .leftWrist, .rightWrist,
        .leftHip, .rightHip,
        .leftKnee, .rightKnee,
        .leftAnkle, .rightAnkle
    ]

    // Internal for testing: COCO17→COCO18 append + OpenPose reindexing.
    static func reorder(coco17: [Keypoint], neck: Keypoint) -> [Keypoint] {
        let coco18Raw = coco17 + [neck]
        return oppOrder.map { coco18Raw[$0] }
    }

    // Internal for testing: selects neck from Vision (if conf ≥ 0.3) or shoulder average.
    static func selectNeck(
        visionNeck: Keypoint?,
        leftShoulder: Keypoint,
        rightShoulder: Keypoint
    ) -> Keypoint {
        if let neck = visionNeck, neck.confidence >= 0.3 {
            return neck
        }
        return Keypoint(
            x: (leftShoulder.x + rightShoulder.x) * 0.5,
            y: (leftShoulder.y + rightShoulder.y) * 0.5,
            confidence: (leftShoulder.confidence + rightShoulder.confidence) * 0.5
        )
    }

    /// Converts a pose observation to an 18-keypoint skeleton in normalized coordinates (0–1).
    ///
    /// Vision already delivers points in normalized space with (0,0) at bottom-left.
    /// This method flips y so (0,0) is top-left (UIKit/SwiftUI convention) and reindexes
    /// to COCO18/OpenPose order. No pixel dimensions are needed.
    ///
    /// - Parameters:
    ///   - observation: Vision pose observation.
    ///   - frameIndex: Current frame counter.
    ///   - timestamp: Current frame timestamp.
    func convert(
        _ observation: VNHumanBodyPoseObservation,
        frameIndex: Int,
        timestamp: CMTime
    ) throws -> PoseSkeleton {
        var coco17 = [Keypoint](repeating: Keypoint(x: 0, y: 0, confidence: 0), count: 17)

        for (i, jointName) in Self.coco17Joints.enumerated() {
            if let point = try? observation.recognizedPoint(jointName), point.confidence > 0 {
                // Vision gives (0,0)=bottom-left in normalized coords; flip y for UIKit.
                coco17[i] = Keypoint(
                    x: Float(point.location.x),
                    y: Float(1.0 - point.location.y),
                    confidence: Float(point.confidence)
                )
            }
        }

        // Build neck: use Vision's neck joint if confidence >= 0.3, else average shoulders.
        let neckKeypoint: Keypoint
        if let neckPoint = try? observation.recognizedPoint(.neck), neckPoint.confidence >= 0.3 {
            neckKeypoint = Keypoint(
                x: Float(neckPoint.location.x),
                y: Float(1.0 - neckPoint.location.y),
                confidence: Float(neckPoint.confidence)
            )
        } else {
            let ls = coco17[5]
            let rs = coco17[6]
            neckKeypoint = Keypoint(
                x: (ls.x + rs.x) * 0.5,
                y: (ls.y + rs.y) * 0.5,
                confidence: (ls.confidence + rs.confidence) * 0.5
            )
        }

        // coco18: append neck at index 17, then reorder via oppOrder.
        let coco18Raw = coco17 + [neckKeypoint]  // indices 0..17
        let reordered = Self.oppOrder.map { coco18Raw[$0] }

        // Bounding box in normalized (0–1) space.
        let valid = reordered.filter { $0.confidence > 0 }
        let bbox: CGRect
        if valid.isEmpty {
            bbox = .zero
        } else {
            let xs = valid.map { CGFloat($0.x) }
            let ys = valid.map { CGFloat($0.y) }
            let minX = xs.min()!, maxX = xs.max()!
            let minY = ys.min()!, maxY = ys.max()!
            bbox = CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
        }

        return PoseSkeleton(
            keypoints: reordered,
            frameIndex: frameIndex,
            timestamp: timestamp,
            boundingBox: bbox
        )
    }
}
