import Foundation

/// Thread-safe 24-frame rolling window for pose skeletons (Swift actor).
actor FrameBuffer {
    static let capacity = 24
    private var frames: [PoseSkeleton] = []

    var isReady: Bool { frames.count == Self.capacity }
    var count: Int { frames.count }

    func append(_ skeleton: PoseSkeleton) {
        if frames.count == Self.capacity {
            frames.removeFirst()
        }
        frames.append(skeleton)
    }

    func currentWindow() -> [PoseSkeleton]? {
        guard isReady else { return nil }
        return frames
    }

    func reset() {
        frames.removeAll()
    }
}
