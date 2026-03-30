import XCTest
import CoreMedia
@testable import ShopliftDetect

@MainActor
final class TrackingServiceTests: XCTestCase {
    var sut: TrackingService!

    override func setUp() {
        sut = TrackingService()
    }

    // MARK: - Helpers

    private func skeleton(box: CGRect) -> PoseSkeleton {
        PoseSkeleton(
            keypoints: [],
            frameIndex: 0,
            timestamp: CMTime.zero,
            boundingBox: box
        )
    }

    // MARK: - Tests

    func test_firstSkeleton_returnsNewID() {
        let id = sut.matchTrack(for: skeleton(box: CGRect(x: 0.1, y: 0.1, width: 0.2, height: 0.4)))
        XCTAssertFalse(id.isEmpty)
    }

    func test_overlappingBoxes_returnsSameTrackID() {
        let box = CGRect(x: 0.1, y: 0.1, width: 0.3, height: 0.5)
        let id1 = sut.matchTrack(for: skeleton(box: box))
        // Slightly shifted but heavily overlapping
        let shifted = CGRect(x: 0.11, y: 0.11, width: 0.3, height: 0.5)
        let id2 = sut.matchTrack(for: skeleton(box: shifted))
        XCTAssertEqual(id1, id2)
    }

    func test_disjointBoxes_returnsDifferentTrackIDs() {
        let id1 = sut.matchTrack(for: skeleton(box: CGRect(x: 0.0, y: 0.0, width: 0.1, height: 0.1)))
        let id2 = sut.matchTrack(for: skeleton(box: CGRect(x: 0.9, y: 0.9, width: 0.1, height: 0.1)))
        XCTAssertNotEqual(id1, id2)
    }

    func test_iouExactlyAtThreshold_returnsSameID() {
        // Build two boxes whose IoU is exactly 0.3.
        // Box A: (0,0,1,1) area=1. Box B: (x,0,1,1).
        // intersection width = 1-x, area = 1-x. union = 1 + 1 - (1-x) = 1+x.
        // IoU = (1-x)/(1+x) = 0.3  =>  1-x = 0.3+0.3x  =>  0.7x = 0.7  => x = 0.7/1.3
        let x = 0.7 / 1.3
        let boxA = CGRect(x: 0, y: 0, width: 1, height: 1)
        let boxB = CGRect(x: x, y: 0, width: 1, height: 1)
        let id1 = sut.matchTrack(for: skeleton(box: boxA))
        let id2 = sut.matchTrack(for: skeleton(box: boxB))
        XCTAssertEqual(id1, id2)
    }

    func test_belowThreshold_returnsNewID() {
        // IoU just below 0.3 — new track expected.
        let x = 0.7 / 1.3 + 0.01  // slightly above boundary → IoU < 0.3
        let boxA = CGRect(x: 0, y: 0, width: 1, height: 1)
        let boxB = CGRect(x: x, y: 0, width: 1, height: 1)
        let id1 = sut.matchTrack(for: skeleton(box: boxA))
        let id2 = sut.matchTrack(for: skeleton(box: boxB))
        XCTAssertNotEqual(id1, id2)
    }

    func test_multipleCalls_sameBoxReturnsConsistentID() {
        let box = CGRect(x: 0.2, y: 0.2, width: 0.3, height: 0.4)
        let id1 = sut.matchTrack(for: skeleton(box: box))
        let id2 = sut.matchTrack(for: skeleton(box: box))
        let id3 = sut.matchTrack(for: skeleton(box: box))
        XCTAssertEqual(id1, id2)
        XCTAssertEqual(id2, id3)
    }
}
