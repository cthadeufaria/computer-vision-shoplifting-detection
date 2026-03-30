import SwiftUI
import AVFoundation
import Combine
import CoreMedia
import UIKit

@MainActor
final class DetectionViewModel: ObservableObject {
    @Published var detectionState: DetectionState = .idle
    @Published var skeletons: [PoseSkeleton] = []

    private let cameraSession = CameraSession()
    private let poseEstimator = PoseEstimator()
    private let keypointConverter = KeypointConverter()
    private let anomalyScorer = AnomalyScorer()
    private var modelRunner: STGNFModelRunner?

    // Per-track frame buffers keyed by IoU-matched bounding box track IDs.
    private var trackBuffers: [String: FrameBuffer] = [:]
    private var frameIndex = 0
    private var cancellables = Set<AnyCancellable>()

    var previewLayer: AVCaptureVideoPreviewLayer { cameraSession.previewLayer }

    func enablePreviewTestMode() {
        detectionState = .warmingUp(framesCollected: 0, framesNeeded: FrameBuffer.capacity)
        skeletons = []
    }

    func start() throws {
        modelRunner = try? STGNFModelRunner()
        try cameraSession.start()
        cameraSession.framePublisher
            .sink { [weak self] pixelBuffer in
                guard let self else { return }
                Task { [weak self] in
                    await self?.processFrame(pixelBuffer)
                }
            }
            .store(in: &cancellables)
        detectionState = .warmingUp(framesCollected: 0, framesNeeded: FrameBuffer.capacity)
    }

    func stop() {
        cameraSession.stop()
        cancellables.removeAll()
        trackBuffers.removeAll()
        frameIndex = 0
        detectionState = .idle
    }

    // nonisolated so Vision and CoreML run on the cooperative thread pool,
    // never on the main thread. MainActor state is accessed via await MainActor.run.
    nonisolated private func processFrame(_ pixelBuffer: CVPixelBuffer) async {
        // Snapshot the Sendable objects we need off MainActor.
        let snapshot: (Int, PoseEstimator, KeypointConverter, AnomalyScorer, STGNFModelRunner?, UIDeviceOrientation)? = await MainActor.run { [weak self] in
            guard let self else { return nil }
            return (self.frameIndex, self.poseEstimator, self.keypointConverter, self.anomalyScorer,
                    self.modelRunner, UIDevice.current.orientation)
        }
        guard let (currentFrameIndex, estimator, converter, scorer, runner, deviceOrientation) = snapshot else { return }

        // Vision: runs on cooperative thread pool (NOT main thread).
        guard let observations = try? estimator.detectPoses(in: pixelBuffer,
                                                            deviceOrientation: deviceOrientation) else { return }

        let now = CMTime(seconds: Date().timeIntervalSince1970, preferredTimescale: 600)

        var currentSkeletons: [PoseSkeleton] = []

        for observation in observations {
            guard let skeleton = try? converter.convert(
                observation,
                frameIndex: currentFrameIndex,
                timestamp: now
            ) else { continue }
            currentSkeletons.append(skeleton)

            // matchTrack mutates MainActor-isolated state — run on MainActor.
            let buffer: FrameBuffer = await MainActor.run { [weak self] in
                guard let self else { return FrameBuffer() }
                let trackID = matchTrack(for: skeleton)
                let buf = trackBuffers[trackID, default: FrameBuffer()]
                trackBuffers[trackID] = buf
                return buf
            }

            await buffer.append(skeleton)

            if await buffer.isReady, let window = await buffer.currentWindow() {
                let normalizer = PoseNormalizer()
                // CoreML: also on cooperative thread pool (NOT main thread).
                if let mlArray = try? normalizer.normalize(window),
                   let runner,
                   let score = try? runner.runInference(on: mlArray) {
                    let result = scorer.classify(score: score, isWarmup: false)
                    await MainActor.run { [weak self] in
                        self?.detectionState = .running(latestResult: result)
                    }
                }
            }
        }

        // Commit UI updates.
        await MainActor.run { [weak self] in
            guard let self else { return }
            skeletons = currentSkeletons
            frameIndex += 1
        }

        // Warmup counter update (FrameBuffer.count is async — actor isolated).
        let firstBuffer: FrameBuffer? = await MainActor.run { [weak self] in
            self?.trackBuffers.values.first
        }
        if let firstBuffer {
            let count = await firstBuffer.count
            await MainActor.run { [weak self] in
                guard let self, case .warmingUp = detectionState else { return }
                detectionState = .warmingUp(framesCollected: count, framesNeeded: FrameBuffer.capacity)
            }
        }
    }

    // Called only from MainActor context (processFrame routes through MainActor.run).
    private func matchTrack(for skeleton: PoseSkeleton) -> String {
        let iouThreshold: CGFloat = 0.3
        var bestID: String?
        var bestIoU: CGFloat = 0

        for (id, _) in trackBuffers {
            if let iou = lastBBox[id].map({ computeIoU(skeleton.boundingBox, $0) }),
               iou > bestIoU {
                bestIoU = iou
                bestID = id
            }
        }

        if let id = bestID, bestIoU >= iouThreshold {
            lastBBox[id] = skeleton.boundingBox
            return id
        }

        let newID = UUID().uuidString
        lastBBox[newID] = skeleton.boundingBox
        return newID
    }

    private var lastBBox: [String: CGRect] = [:]

    private func computeIoU(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let intersection = a.intersection(b)
        guard !intersection.isNull else { return 0 }
        let intersectionArea = intersection.width * intersection.height
        let unionArea = a.width * a.height + b.width * b.height - intersectionArea
        return unionArea > 0 ? intersectionArea / unionArea : 0
    }
}
