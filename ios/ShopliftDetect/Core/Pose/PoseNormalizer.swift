import CoreML

/// Normalizes a 24-frame pose window and converts to MLMultiArray [1, 2, 24, 18].
/// Exact port of data_utils.py:normalize_pose (live branch).
struct PoseNormalizer {
    let videoWidth: Float
    let videoHeight: Float

    /// - Parameter window: 24 PoseSkeleton frames, each with 18 keypoints.
    /// - Returns: MLMultiArray of shape [1, 2, 24, 18] (Float32), conf channel dropped.
    func normalize(_ window: [PoseSkeleton]) throws -> MLMultiArray {
        precondition(window.count == 24, "Window must contain exactly 24 frames")

        let numFrames = 24
        let numJoints = 18

        // Step 1: copy xy into [24, 18, 2], dividing by resolution.
        var xy = [[[Float]]](
            repeating: [[Float]](repeating: [Float](repeating: 0, count: 2), count: numJoints),
            count: numFrames
        )
        for (f, skeleton) in window.enumerated() {
            for (j, kp) in skeleton.keypoints.enumerated() {
                xy[f][j][0] = kp.x / videoWidth
                xy[f][j][1] = kp.y / videoHeight
            }
        }

        // Step 2: subtract spatial mean over all 24×18 xy values.
        var sumX: Float = 0, sumY: Float = 0
        let count = Float(numFrames * numJoints)
        for f in 0..<numFrames {
            for j in 0..<numJoints {
                sumX += xy[f][j][0]
                sumY += xy[f][j][1]
            }
        }
        let meanX = sumX / count
        let meanY = sumY / count
        for f in 0..<numFrames {
            for j in 0..<numJoints {
                xy[f][j][0] -= meanX
                xy[f][j][1] -= meanY
            }
        }

        // Step 3: compute std on y-column only (all 24×18 y values after mean subtraction).
        var sumSqY: Float = 0
        for f in 0..<numFrames {
            for j in 0..<numJoints {
                sumSqY += xy[f][j][1] * xy[f][j][1]
            }
        }
        let varY = sumSqY / count
        let stdY = varY > 0 ? sqrtf(varY) : 1.0  // guard against zero variance

        // Step 4: divide both x and y by stdY.
        for f in 0..<numFrames {
            for j in 0..<numJoints {
                xy[f][j][0] /= stdY
                xy[f][j][1] /= stdY
            }
        }

        // Step 5: build MLMultiArray [1, 2, 24, 18] — channel 0 = x, channel 1 = y.
        let shape: [NSNumber] = [1, 2, 24, 18]
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        for f in 0..<numFrames {
            for j in 0..<numJoints {
                // index = batch*2*24*18 + channel*24*18 + frame*18 + joint
                let xIdx = 0 * numFrames * numJoints + f * numJoints + j
                let yIdx = 1 * numFrames * numJoints + f * numJoints + j
                array[xIdx] = NSNumber(value: xy[f][j][0])
                array[yIdx] = NSNumber(value: xy[f][j][1])
            }
        }

        return array
    }
}
