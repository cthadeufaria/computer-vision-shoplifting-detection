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

    /// Converts a pose observation to an 18-keypoint skeleton in pixel coordinates.
    /// - Parameters:
    ///   - observation: Vision pose observation.
    ///   - previewSize: Pixel dimensions of the camera preview (width × height).
    ///   - frameIndex: Current frame counter.
    ///   - timestamp: Current frame timestamp.
    func convert(
        _ observation: VNHumanBodyPoseObservation,
        previewSize: CGSize,
        frameIndex: Int,
        timestamp: CMTime
    ) throws -> PoseSkeleton {
        var coco17 = [Keypoint](repeating: Keypoint(x: 0, y: 0, confidence: 0), count: 17)

        for (i, jointName) in Self.coco17Joints.enumerated() {
            if let point = try? observation.recognizedPoint(jointName), point.confidence > 0 {
                // Vision gives (0,0)=bottom-left in normalized coords; flip y for UIKit.
                let px = Float(point.location.x) * Float(previewSize.width)
                let py = Float(1.0 - point.location.y) * Float(previewSize.height)
                coco17[i] = Keypoint(x: px, y: py, confidence: Float(point.confidence))
            }
        }

        // Build neck: use Vision's neck joint if confidence >= 0.3, else average shoulders.
        let neckKeypoint: Keypoint
        if let neckPoint = try? observation.recognizedPoint(.neck), neckPoint.confidence >= 0.3 {
            let px = Float(neckPoint.location.x) * Float(previewSize.width)
            let py = Float(1.0 - neckPoint.location.y) * Float(previewSize.height)
            neckKeypoint = Keypoint(x: px, y: py, confidence: Float(neckPoint.confidence))
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
        var coco18Raw = coco17 + [neckKeypoint]  // indices 0..17
        let reordered = Self.oppOrder.map { coco18Raw[$0] }

        // Bounding box from all non-zero-confidence keypoints.
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
