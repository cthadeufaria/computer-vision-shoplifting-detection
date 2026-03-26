import XCTest
import CoreML
import CoreMedia
@testable import ShopliftDetect

final class PoseNormalizerTests: XCTestCase {

    // MARK: - Helpers

    /// Creates a 24-frame window with varying keypoints in normalized (0–1) coordinates.
    private func makeSyntheticWindow(confidence: Float = 0.8) -> [PoseSkeleton] {
        (0..<24).map { t in
            PoseSkeleton(
                keypoints: (0..<18).map { j in
                    Keypoint(
                        x: Float(t * 20 + j * 5 + 10) / 1000,   // 0.01–0.49, distinct per (t,j)
                        y: Float(t * 15 + j * 10 + 5) / 1000,   // 0.005–0.37, distinct per (t,j)
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

    func testMeanSubtractionOverAllFramesAndJoints() throws {
        let skeletons = makeSyntheticWindow()
        let array = try PoseNormalizer().normalize(skeletons)
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
        let array = try PoseNormalizer().normalize(skeletons)
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
        let array1 = try PoseNormalizer().normalize(skeletons1)
        let array2 = try PoseNormalizer().normalize(skeletons2)
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

        // The fixture stores pixel-space coordinates. Pre-divide by vid_res to produce the
        // same 0–1 inputs that KeypointConverter now delivers at runtime.
        let skeletons: [PoseSkeleton] = (0..<24).map { t in
            PoseSkeleton(
                keypoints: (0..<18).map { j in
                    Keypoint(
                        x: Float(inputRaw[0][t][j][0]) / Float(vidRes[0]),
                        y: Float(inputRaw[0][t][j][1]) / Float(vidRes[1]),
                        confidence: Float(inputRaw[0][t][j][2])
                    )
                },
                frameIndex: t,
                timestamp: .zero,
                boundingBox: .zero
            )
        }

        let array = try PoseNormalizer().normalize(skeletons)

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
                keypoints: Array(repeating: Keypoint(x: 0.5, y: 0.5, confidence: 0.8), count: 18),
                frameIndex: t, timestamp: .zero, boundingBox: .zero
            )
        }
        let array = try PoseNormalizer().normalize(skeletons)
        for i in 0..<(2 * 24 * 18) {
            XCTAssertFalse(array[i].floatValue.isNaN, "NaN at index \(i)")
            XCTAssertFalse(array[i].floatValue.isInfinite, "Inf at index \(i)")
        }
    }

    func testOutputShapeIs_1_2_24_18() throws {
        let skeletons = makeSyntheticWindow()
        let array = try PoseNormalizer().normalize(skeletons)
        XCTAssertEqual(array.shape.map { $0.intValue }, [1, 2, 24, 18])
    }
}
