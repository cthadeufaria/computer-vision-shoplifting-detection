import CoreImage
import CoreMedia
import UIKit

actor RemoteInferenceService {
    private let estimator: any PoseEstimatorProtocol
    private let converter: any KeypointConverterProtocol
    private let scorerThreshold: Float
    private let ciContext = CIContext()
    private var frameIndexBySession: [UUID: Int] = [:]
    private var frameBuffers: [String: FrameBuffer] = [:]
    private var modelRunner: STGNFModelRunner?

    init(
        estimator: any PoseEstimatorProtocol,
        converter: any KeypointConverterProtocol,
        scorerThreshold: Float
    ) {
        self.estimator = estimator
        self.converter = converter
        self.scorerThreshold = scorerThreshold
        self.modelRunner = try? STGNFModelRunner()
    }

    func inferDetections(for frame: VideoFrame, sessionID: UUID) async -> [DetectionResult] {
        guard let pixelBuffer = makePixelBuffer(from: frame) else { return [] }

        let frameIndex = frameIndexBySession[sessionID, default: 0]
        frameIndexBySession[sessionID] = frameIndex + 1

        guard let observations = try? estimator.detectPoses(in: pixelBuffer, deviceOrientation: .portrait) else {
            return []
        }

        let timestamp = CMTime(seconds: Date().timeIntervalSince1970, preferredTimescale: 600)
        var detections: [DetectionResult] = []

        for (index, observation) in observations.enumerated() {
            guard let skeleton = try? converter.convert(
                observation,
                frameIndex: frameIndex,
                timestamp: timestamp
            ) else {
                continue
            }

            let trackKey = "\(sessionID.uuidString)-\(index)"
            let buffer = frameBuffers[trackKey, default: FrameBuffer()]
            frameBuffers[trackKey] = buffer
            await buffer.append(skeleton)

            let scorer = AnomalyScorer(threshold: scorerThreshold)
            let result: AnomalyResult
            if await buffer.isReady,
               let window = await buffer.currentWindow(),
               let mlArray = try? PoseNormalizer().normalize(window),
               let modelRunner,
               let score = try? modelRunner.runInference(on: mlArray) {
                result = scorer.classify(score: score, isWarmup: false)
            } else {
                result = scorer.classify(score: 0, isWarmup: true)
            }

            detections.append(
                DetectionResult(
                    trackID: index + 1,
                    score: result.score,
                    label: result.label,
                    keypoints: skeleton.keypoints,
                    boundingBox: skeleton.boundingBox,
                    timestamp: result.timestamp
                )
            )
        }

        return detections
    }

    private func makePixelBuffer(from frame: VideoFrame) -> CVPixelBuffer? {
        guard let image = CIImage(data: frame.jpegData) else { return nil }

        var pixelBuffer: CVPixelBuffer?
        let attributes = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ] as CFDictionary

        let width = max(frame.width, Int(image.extent.width))
        let height = max(frame.height, Int(image.extent.height))
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attributes,
            &pixelBuffer
        )
        guard status == kCVReturnSuccess, let pixelBuffer else { return nil }

        ciContext.render(image, to: pixelBuffer)
        return pixelBuffer
    }
}
