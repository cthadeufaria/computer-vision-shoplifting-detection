import Vision
import CoreMedia
@testable import ShopliftDetect

final class MockKeypointConverter: KeypointConverterProtocol, @unchecked Sendable {
    var stubbedSkeleton: PoseSkeleton?
    var shouldThrow = false
    var convertCallCount = 0

    func convert(
        _ observation: VNHumanBodyPoseObservation,
        frameIndex: Int,
        timestamp: CMTime
    ) throws -> PoseSkeleton {
        convertCallCount += 1
        if shouldThrow { throw MockError.generic }
        return stubbedSkeleton ?? PoseSkeleton(
            keypoints: [],
            frameIndex: frameIndex,
            timestamp: timestamp,
            boundingBox: .zero
        )
    }
}
