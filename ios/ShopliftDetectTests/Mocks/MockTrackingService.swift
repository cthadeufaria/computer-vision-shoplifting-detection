import CoreMedia
@testable import ShopliftDetect

final class MockTrackingService: TrackingServiceProtocol {
    var stubbedTrackID = "mock-track-id"
    var matchCallCount = 0

    func matchTrack(for skeleton: PoseSkeleton) -> String {
        matchCallCount += 1
        return stubbedTrackID
    }
}
