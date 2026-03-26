import CoreMedia

/// 18 keypoints in OpenPose COCO18 order (after COCO17→COCO18 conversion + reindexing).
struct PoseSkeleton: Sendable {
    let keypoints: [Keypoint]
    let frameIndex: Int
    let timestamp: CMTime

    /// Bounding box in normalized coordinates (0–1), used for IoU-based person tracking.
    let boundingBox: CGRect
}
