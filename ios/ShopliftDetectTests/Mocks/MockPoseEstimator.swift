import Vision
import UIKit
@testable import ShopliftDetect

final class MockPoseEstimator: PoseEstimatorProtocol, @unchecked Sendable {
    var stubbedObservations: [VNHumanBodyPoseObservation] = []
    var shouldThrow = false
    var detectCallCount = 0

    func detectPoses(
        in pixelBuffer: CVPixelBuffer,
        deviceOrientation: UIDeviceOrientation
    ) throws -> [VNHumanBodyPoseObservation] {
        detectCallCount += 1
        if shouldThrow { throw MockError.generic }
        return stubbedObservations
    }
}

enum MockError: Error { case generic }
