import XCTest
@testable import ShopliftDetect

final class FrameBufferTests: XCTestCase {

    private func makeSkeleton(_ index: Int) -> PoseSkeleton {
        PoseSkeleton(
            keypoints: Array(repeating: Keypoint(x: 0, y: 0, confidence: 0), count: 18),
            frameIndex: index,
            timestamp: .zero,
            boundingBox: .zero
        )
    }

    func testEmptyBufferIsNotReady() async {
        let buffer = FrameBuffer()
        let ready = await buffer.isReady
        XCTAssertFalse(ready)
    }

    func testPartialBufferIsNotReadyBelow24() async {
        let buffer = FrameBuffer()
        for i in 0..<23 { await buffer.append(makeSkeleton(i)) }
        let ready = await buffer.isReady
        XCTAssertFalse(ready)
    }

    func testAt24FramesBufferIsReady() async {
        let buffer = FrameBuffer()
        for i in 0..<24 { await buffer.append(makeSkeleton(i)) }
        let ready = await buffer.isReady
        XCTAssertTrue(ready)
    }

    func testFrame25EvintsOldestFrame() async {
        let buffer = FrameBuffer()
        for i in 0..<24 { await buffer.append(makeSkeleton(i)) }
        await buffer.append(makeSkeleton(24))
        let window = await buffer.currentWindow()
        XCTAssertEqual(window?.first?.frameIndex, 1)
    }

    func testFIFOOrderingPreserved() async {
        let buffer = FrameBuffer()
        for i in 0..<24 { await buffer.append(makeSkeleton(i)) }
        let window = await buffer.currentWindow()
        let indices = window?.map { $0.frameIndex }
        XCTAssertEqual(indices, Array(0..<24))
    }

    func testResetClearsFrames() async {
        let buffer = FrameBuffer()
        for i in 0..<24 { await buffer.append(makeSkeleton(i)) }
        await buffer.reset()
        let count = await buffer.count
        XCTAssertEqual(count, 0)
    }

    func testExportedTensorShapeIs_2_24_18() async throws {
        let buffer = FrameBuffer()
        for i in 0..<24 { await buffer.append(makeSkeleton(i)) }
        let window = await buffer.currentWindow()
        XCTAssertNotNil(window)
        let normalizer = PoseNormalizer(videoWidth: 640, videoHeight: 480)
        let array = try normalizer.normalize(window!)
        XCTAssertEqual(array.shape, [1, 2, 24, 18])
    }

    func testConcurrentAccessIsThreadSafe() async {
        let buffer = FrameBuffer()
        // Append 48 frames concurrently — should not crash and buffer should be consistent.
        await withTaskGroup(of: Void.self) { group in
            for i in 0..<48 {
                let skeleton = self.makeSkeleton(i)
                group.addTask { await buffer.append(skeleton) }
            }
        }
        let count = await buffer.count
        XCTAssertEqual(count, 24)
    }
}
