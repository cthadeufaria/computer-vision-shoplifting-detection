import Vision
import UIKit

protocol PoseEstimatorProtocol: Sendable {
    func detectPoses(
        in pixelBuffer: CVPixelBuffer,
        deviceOrientation: UIDeviceOrientation
    ) throws -> [VNHumanBodyPoseObservation]
}
