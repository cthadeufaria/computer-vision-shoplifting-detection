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

    private func skeleton(box: CGRect, frameIndex: Int = 0) -> PoseSkeleton {
        PoseSkeleton(
            keypoints: [],
            frameIndex: frameIndex,
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
        let id1 = sut.matchTrack(for: skeleton(box: box, frameIndex: 0))
        // Slightly shifted but heavily overlapping
        let shifted = CGRect(x: 0.11, y: 0.11, width: 0.3, height: 0.5)
        let id2 = sut.matchTrack(for: skeleton(box: shifted, frameIndex: 1))
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
        let id1 = sut.matchTrack(for: skeleton(box: boxA, frameIndex: 0))
        let id2 = sut.matchTrack(for: skeleton(box: boxB, frameIndex: 1))
        XCTAssertEqual(id1, id2)
    }

    func test_belowThreshold_returnsNewID() {
        // IoU just below 0.3 — new track expected.
        let x = 0.7 / 1.3 + 0.01  // slightly above boundary → IoU < 0.3
        let boxA = CGRect(x: 0, y: 0, width: 1, height: 1)
        let boxB = CGRect(x: x, y: 0, width: 1, height: 1)
        let id1 = sut.matchTrack(for: skeleton(box: boxA, frameIndex: 0))
        let id2 = sut.matchTrack(for: skeleton(box: boxB, frameIndex: 1))
        XCTAssertNotEqual(id1, id2)
    }

    func test_multipleCalls_sameBoxReturnsConsistentID() {
        let box = CGRect(x: 0.2, y: 0.2, width: 0.3, height: 0.4)
        let id1 = sut.matchTrack(for: skeleton(box: box, frameIndex: 0))
        let id2 = sut.matchTrack(for: skeleton(box: box, frameIndex: 1))
        let id3 = sut.matchTrack(for: skeleton(box: box, frameIndex: 2))
        XCTAssertEqual(id1, id2)
        XCTAssertEqual(id2, id3)
    }

    func test_sameFrameDifferentPeople_doNotReuseTrackID() {
        let first = PoseSkeleton(
            keypoints: [],
            frameIndex: 10,
            timestamp: .zero,
            boundingBox: CGRect(x: 0.1, y: 0.1, width: 0.2, height: 0.4)
        )
        let second = PoseSkeleton(
            keypoints: [],
            frameIndex: 10,
            timestamp: .zero,
            boundingBox: CGRect(x: 0.12, y: 0.1, width: 0.2, height: 0.4)
        )

        let id1 = sut.matchTrack(for: first)
        let id2 = sut.matchTrack(for: second)

        XCTAssertNotEqual(id1, id2)
    }

    func test_trackReusedWithinMaxMissingWindow() {
        let original = PoseSkeleton(
            keypoints: [],
            frameIndex: 0,
            timestamp: .zero,
            boundingBox: CGRect(x: 0.2, y: 0.2, width: 0.2, height: 0.3)
        )
        let returned = PoseSkeleton(
            keypoints: [],
            frameIndex: 6,
            timestamp: .zero,
            boundingBox: CGRect(x: 0.21, y: 0.2, width: 0.2, height: 0.3)
        )

        let originalID = sut.matchTrack(for: original)
        let returnedID = sut.matchTrack(for: returned)

        XCTAssertEqual(originalID, returnedID)
    }

    func test_trackExpiresAfterMaxMissingFrames() {
        let original = PoseSkeleton(
            keypoints: [],
            frameIndex: 0,
            timestamp: .zero,
            boundingBox: CGRect(x: 0.2, y: 0.2, width: 0.2, height: 0.3)
        )
        let returned = PoseSkeleton(
            keypoints: [],
            frameIndex: 7,
            timestamp: .zero,
            boundingBox: CGRect(x: 0.2, y: 0.2, width: 0.2, height: 0.3)
        )

        let originalID = sut.matchTrack(for: original)
        let returnedID = sut.matchTrack(for: returned)

        XCTAssertNotEqual(originalID, returnedID)
    }
}
