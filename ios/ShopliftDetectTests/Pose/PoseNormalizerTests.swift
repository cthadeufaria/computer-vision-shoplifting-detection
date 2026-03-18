import XCTest
import CoreML
import CoreMedia
@testable import ShopliftDetect

final class PoseNormalizerTests: XCTestCase {

    // MARK: - Helpers

    /// Creates a 24-frame window with varying, non-constant keypoint positions.
    private func makeSyntheticWindow(confidence: Float = 0.8) -> [PoseSkeleton] {
        (0..<24).map { t in
            PoseSkeleton(
                keypoints: (0..<18).map { j in
                    Keypoint(
                        x: Float(t * 20 + j * 5 + 10),    // 10..490, distinct per (t,j)
                        y: Float(t * 15 + j * 10 + 5),    // 5..370, distinct per (t,j)
                        confidence: confidence
                    )
                },
                frameIndex: t,
                timestamp: .zero,
                boundingBox: .zero
            )
        }
    }

    private func loadFixtureData(_ name: String) throws -> Data {
        if let url = Bundle(for: Self.self).url(forResource: name, withExtension: "json") {
            return try Data(contentsOf: url)
        }
        let sourceFile = URL(fileURLWithPath: #filePath)
        let fixturesDir = sourceFile
            .deletingLastPathComponent()  // .../Pose/
            .deletingLastPathComponent()  // .../ShopliftDetectTests/
            .appendingPathComponent("Fixtures")
        return try Data(contentsOf: fixturesDir.appendingPathComponent("\(name).json"))
    }

    // MARK: - Tests

    func testResolutionDivisionByWidthAndHeight() throws {
        // All joints at (0,0) except frame 0, joint 0 at (640, 0).
        // After div by W=640: x[0,0]=1.0. After mean sub & stdY-guard=1.0: x[0,0] = 431/432.
        // If x were incorrectly divided by H=480: x[0,0] = 640/480*(431/432) ≈ 1.33.
        let W: Float = 640, H: Float = 480
        let skeletons = (0..<24).map { t in
            PoseSkeleton(
                keypoints: (0..<18).map { j in
                    Keypoint(x: (t == 0 && j == 0) ? W : 0, y: 0, confidence: 0.8)
                },
                frameIndex: t, timestamp: .zero, boundingBox: .zero
            )
        }
        let array = try PoseNormalizer(videoWidth: W, videoHeight: H).normalize(skeletons)
        let expected: Float = 431.0 / 432.0  // (1 - 1/432) / stdY=1.0
        XCTAssertEqual(array[0].floatValue, expected, accuracy: 1e-4)
        // Sanity: if W/H were swapped the value would be ~1.33, not ~0.998.
        XCTAssertLessThan(array[0].floatValue, 1.1)
    }

    func testMeanSubtractionOverAllFramesAndJoints() throws {
        let skeletons = makeSyntheticWindow()
        let array = try PoseNormalizer(videoWidth: 640, videoHeight: 480).normalize(skeletons)
        var sumX: Float = 0, sumY: Float = 0
        for t in 0..<24 {
            for j in 0..<18 {
                sumX += array[t * 18 + j].floatValue
                sumY += array[432 + t * 18 + j].floatValue
            }
        }
        XCTAssertEqual(sumX / 432, 0, accuracy: 1e-5)
        XCTAssertEqual(sumY / 432, 0, accuracy: 1e-5)
    }

    func testYAxisStdAppliedToBothXandY() throws {
        let skeletons = makeSyntheticWindow()
        let array = try PoseNormalizer(videoWidth: 640, videoHeight: 480).normalize(skeletons)
        var sumSqY: Float = 0
        for t in 0..<24 {
            for j in 0..<18 {
                let y = array[432 + t * 18 + j].floatValue
                sumSqY += y * y
            }
        }
        let stdY = sqrtf(sumSqY / 432)
        XCTAssertEqual(stdY, 1.0, accuracy: 1e-4)
    }

    func testConfidencePassesThroughUnmodified() throws {
        // Two windows with identical xy but different confidence; xy output must be identical.
        let skeletons1 = makeSyntheticWindow(confidence: 0.5)
        let skeletons2 = makeSyntheticWindow(confidence: 0.9)
        let array1 = try PoseNormalizer(videoWidth: 640, videoHeight: 480).normalize(skeletons1)
        let array2 = try PoseNormalizer(videoWidth: 640, videoHeight: 480).normalize(skeletons2)
        var maxDiff: Float = 0
        for i in 0..<(2 * 24 * 18) {
            maxDiff = max(maxDiff, abs(array1[i].floatValue - array2[i].floatValue))
        }
        XCTAssertLessThan(maxDiff, 1e-6, "Confidence values should not affect xy normalization")
    }

    func testMatchesPythonOutputForSeededFixture() throws {
        let data = try loadFixtureData("normal_pose_window")
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let inputRaw = json["input"] as! [[[[Double]]]]
        let expectedRaw = json["expected_output"] as! [[[[Double]]]]
        let vidRes = json["vid_res"] as! [Double]

        let skeletons: [PoseSkeleton] = (0..<24).map { t in
            PoseSkeleton(
                keypoints: (0..<18).map { j in
                    Keypoint(
                        x: Float(inputRaw[0][t][j][0]),
                        y: Float(inputRaw[0][t][j][1]),
                        confidence: Float(inputRaw[0][t][j][2])
                    )
                },
                frameIndex: t,
                timestamp: .zero,
                boundingBox: .zero
            )
        }

        let normalizer = PoseNormalizer(videoWidth: Float(vidRes[0]), videoHeight: Float(vidRes[1]))
        let array = try normalizer.normalize(skeletons)

        var maxDiff: Float = 0
        for t in 0..<24 {
            for j in 0..<18 {
                let gotX = array[t * 18 + j].floatValue
                let gotY = array[432 + t * 18 + j].floatValue
                let expX = Float(expectedRaw[0][t][j][0])
                let expY = Float(expectedRaw[0][t][j][1])
                maxDiff = max(maxDiff, abs(gotX - expX), abs(gotY - expY))
            }
        }
        XCTAssertLessThan(maxDiff, 1e-4, "Max abs diff \(maxDiff) vs Python fixture")
    }

    func testZeroVarianceDoesNotProduceNaN() throws {
        // All joints at the same position → stdY = 0 → guard to 1.0 → output all 0, not NaN.
        let skeletons = (0..<24).map { t in
            PoseSkeleton(
                keypoints: Array(repeating: Keypoint(x: 100, y: 200, confidence: 0.8), count: 18),
                frameIndex: t, timestamp: .zero, boundingBox: .zero
            )
        }
        let array = try PoseNormalizer(videoWidth: 640, videoHeight: 480).normalize(skeletons)
        for i in 0..<(2 * 24 * 18) {
            XCTAssertFalse(array[i].floatValue.isNaN, "NaN at index \(i)")
            XCTAssertFalse(array[i].floatValue.isInfinite, "Inf at index \(i)")
        }
    }

    func testOutputShapeIs_1_2_24_18() throws {
        let skeletons = makeSyntheticWindow()
        let array = try PoseNormalizer(videoWidth: 640, videoHeight: 480).normalize(skeletons)
        XCTAssertEqual(array.shape.map { $0.intValue }, [1, 2, 24, 18])
    }

    func testUsesPixelCoordinatesNotNormalized() throws {
        // Pixel input with videoWidth=640 must equal pre-normalized input with videoWidth=1.
        let pixelSkeletons = makeSyntheticWindow()
        let normSkeletons = (0..<24).map { t in
            PoseSkeleton(
                keypoints: (0..<18).map { j in
                    Keypoint(
                        x: Float(t * 20 + j * 5 + 10) / 640.0,
                        y: Float(t * 15 + j * 10 + 5) / 480.0,
                        confidence: 0.8
                    )
                },
                frameIndex: t, timestamp: .zero, boundingBox: .zero
            )
        }
        let pixelOutput = try PoseNormalizer(videoWidth: 640, videoHeight: 480).normalize(pixelSkeletons)
        let normOutput  = try PoseNormalizer(videoWidth: 1,   videoHeight: 1).normalize(normSkeletons)
        var maxDiff: Float = 0
        for i in 0..<(2 * 24 * 18) {
            maxDiff = max(maxDiff, abs(pixelOutput[i].floatValue - normOutput[i].floatValue))
        }
        XCTAssertLessThan(maxDiff, 1e-4)
    }
}
