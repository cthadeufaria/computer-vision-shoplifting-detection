import Foundation
import Vision
import UIKit
@testable import ShopliftDetect

final class MockPoseEstimator: PoseEstimatorProtocol, @unchecked Sendable {
    var stubbedObservations: [VNHumanBodyPoseObservation] = []
    var shouldThrow = false
    var detectCallCount = 0
    var detectionDelayNanoseconds: UInt64 = 0
    private(set) var maxConcurrentCalls = 0

    private let lock = NSLock()
    private var currentConcurrentCalls = 0

    func detectPoses(
        in pixelBuffer: CVPixelBuffer,
        deviceOrientation: UIDeviceOrientation
    ) throws -> [VNHumanBodyPoseObservation] {
        lock.lock()
        detectCallCount += 1
        currentConcurrentCalls += 1
        maxConcurrentCalls = max(maxConcurrentCalls, currentConcurrentCalls)
        lock.unlock()

        defer {
            lock.lock()
            currentConcurrentCalls -= 1
            lock.unlock()
        }

        if detectionDelayNanoseconds > 0 {
            Thread.sleep(forTimeInterval: Double(detectionDelayNanoseconds) / 1_000_000_000)
        }
        if shouldThrow { throw MockError.generic }
        return stubbedObservations
    }
}

enum MockError: Error { case generic }
