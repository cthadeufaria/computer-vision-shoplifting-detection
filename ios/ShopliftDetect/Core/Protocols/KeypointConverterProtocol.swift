import Vision
import CoreMedia

protocol KeypointConverterProtocol: Sendable {
    func convert(
        _ observation: VNHumanBodyPoseObservation,
        frameIndex: Int,
        timestamp: CMTime
    ) throws -> PoseSkeleton
}
