import Vision
import CoreImage

/// Runs VNDetectHumanBodyPoseRequest on camera frames.
final class PoseEstimator: @unchecked Sendable {
    private let requestHandler = VNSequenceRequestHandler()

    func detectPoses(in pixelBuffer: CVPixelBuffer) throws -> [VNHumanBodyPoseObservation] {
        let request = VNDetectHumanBodyPoseRequest()
        try requestHandler.perform([request], on: pixelBuffer, orientation: .up)
        return request.results ?? []
    }
}
