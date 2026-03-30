import XCTest
import Vision
import CoreMedia
@testable import ShopliftDetect

/// Integration test for KeypointConverter.convert(_:frameIndex:timestamp:).
///
/// These tests require a real photograph of a person standing — Vision's neural
/// network must detect at least one pose from the image.
///
/// Fixture: ios/ShopliftDetectTests/Fixtures/person_standing.jpg
/// (or .png — the test accepts either extension)
///
/// If the file is absent or Vision detects no person, every test is skipped
/// automatically so CI does not fail on machines without the fixture.
final class KeypointConverterIntegrationTests: XCTestCase {

    // MARK: - Helpers

    /// Loads the fixture image from source-relative path (same strategy as
    /// PoseNormalizerTests / KeypointConverterTests).
    private func loadFixtureImage() throws -> CGImage? {
        let sourceFile = URL(fileURLWithPath: #filePath)
        let fixturesDir = sourceFile
            .deletingLastPathComponent()  // .../Pose/
            .deletingLastPathComponent()  // .../ShopliftDetectTests/
            .appendingPathComponent("Fixtures")

        for ext in ["jpg", "jpeg", "png"] {
            let url = fixturesDir.appendingPathComponent("person_standing.\(ext)")
            if FileManager.default.fileExists(atPath: url.path),
               let src = CGImageSourceCreateWithURL(url as CFURL, nil),
               let img = CGImageSourceCreateImageAtIndex(src, 0, nil) {
                return img
            }
        }
        return nil
    }

    /// Runs VNDetectHumanBodyPoseRequest on a CGImage and returns all observations.
    private func runVision(on image: CGImage) throws -> [VNHumanBodyPoseObservation] {
        var results: [VNHumanBodyPoseObservation] = []
        let request = VNDetectHumanBodyPoseRequest { req, _ in
            results = req.results as? [VNHumanBodyPoseObservation] ?? []
        }
        try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
        return results
    }

    // MARK: - Fixture set-up (shared across all tests in this class)

    private var observation: VNHumanBodyPoseObservation!

    override func setUpWithError() throws {
        try super.setUpWithError()
        guard let image = try loadFixtureImage() else {
            throw XCTSkip("Fixture image 'person_standing.jpg' not found in Fixtures/ — skipping integration tests")
        }
        let observations = try runVision(on: image)
        guard let first = observations.first else {
            throw XCTSkip("Vision detected no person in fixture image — skipping integration tests")
        }
        observation = first
    }

    // MARK: - Tests

    /// All keypoint x/y values must be in the normalized [0, 1] range.
    /// This is the primary regression guard for the pixel-space bug that was fixed.
    func testAllKeypointsAreNormalized() throws {
        let skeleton = try KeypointConverter().convert(observation, frameIndex: 0, timestamp: .zero)
        for (i, kp) in skeleton.keypoints.enumerated() {
            XCTAssertGreaterThanOrEqual(kp.x, 0.0, "keypoint[\(i)].x below 0: \(kp.x)")
            XCTAssertLessThanOrEqual(kp.x, 1.0, "keypoint[\(i)].x above 1: \(kp.x)")
            XCTAssertGreaterThanOrEqual(kp.y, 0.0, "keypoint[\(i)].y below 0: \(kp.y)")
            XCTAssertLessThanOrEqual(kp.y, 1.0, "keypoint[\(i)].y above 1: \(kp.y)")
        }
    }

    /// Output must always contain exactly 18 keypoints (COCO18/OpenPose format).
    func testOutputHasExactly18Keypoints() throws {
        let skeleton = try KeypointConverter().convert(observation, frameIndex: 0, timestamp: .zero)
        XCTAssertEqual(skeleton.keypoints.count, 18)
    }

    /// Bounding box must fit inside the unit square.
    func testBoundingBoxIsWithinUnitSquare() throws {
        let skeleton = try KeypointConverter().convert(observation, frameIndex: 0, timestamp: .zero)
        let box = skeleton.boundingBox
        XCTAssertGreaterThanOrEqual(box.minX, 0.0, "bbox.minX: \(box.minX)")
        XCTAssertGreaterThanOrEqual(box.minY, 0.0, "bbox.minY: \(box.minY)")
        XCTAssertLessThanOrEqual(box.maxX, 1.0, "bbox.maxX: \(box.maxX)")
        XCTAssertLessThanOrEqual(box.maxY, 1.0, "bbox.maxY: \(box.maxY)")
    }

    /// frameIndex must be passed through unchanged.
    func testFrameIndexPassthrough() throws {
        let skeleton = try KeypointConverter().convert(observation, frameIndex: 42, timestamp: .zero)
        XCTAssertEqual(skeleton.frameIndex, 42)
    }

    /// At least one keypoint should have non-zero confidence (Vision detected something).
    func testAtLeastOneKeypointIsConfident() throws {
        let skeleton = try KeypointConverter().convert(observation, frameIndex: 0, timestamp: .zero)
        let maxConf = skeleton.keypoints.map { $0.confidence }.max() ?? 0
        XCTAssertGreaterThan(maxConf, 0.0, "All keypoints have zero confidence")
    }

    /// y-flip: Vision's (0,0) is bottom-left; after conversion (0,0) must be top-left.
    /// For a standing person the head (nose) y-coordinate should be smaller than the hips.
    /// opp_order[0]=0 → nose; opp_order[8]=12 → leftHip; opp_order[11]=11 → rightHip.
    func testYAxisIsFlippedHeadAboveHips() throws {
        let skeleton = try KeypointConverter().convert(observation, frameIndex: 0, timestamp: .zero)
        let noseY    = skeleton.keypoints[0].y   // OpenPose index 0 = nose
        let leftHipY = skeleton.keypoints[8].y   // OpenPose index 8 = leftHip
        let rightHipY = skeleton.keypoints[11].y // OpenPose index 11 = rightHip
        let noseConf = skeleton.keypoints[0].confidence
        let lhConf   = skeleton.keypoints[8].confidence
        let rhConf   = skeleton.keypoints[11].confidence

        // Only assert when all three landmarks were detected with reasonable confidence.
        guard noseConf > 0.2 && (lhConf > 0.2 || rhConf > 0.2) else {
            throw XCTSkip("Nose or hips not detected with sufficient confidence — skipping y-flip assertion")
        }
        let hipY = lhConf > rhConf ? leftHipY : rightHipY
        XCTAssertLessThan(noseY, hipY, "After y-flip, nose (y=\(noseY)) should be above hips (y=\(hipY))")
    }
}
