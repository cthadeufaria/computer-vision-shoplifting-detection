import SwiftUI
import AVFoundation
import Combine
import CoreMedia
import UIKit

@MainActor
final class DetectionViewModel: ObservableObject {
    @Published var detectionState: DetectionState = .idle
    @Published var skeletons: [PoseSkeleton] = []

    private let camera: CameraSessionProtocol
    private let estimator: any PoseEstimatorProtocol
    private let converter: any KeypointConverterProtocol
    private let scorer: any AnomalyScorerProtocol
    private let tracking: TrackingServiceProtocol
    private var modelRunner: STGNFModelRunner?

    // Per-track frame buffers keyed by IoU-matched bounding box track IDs.
    private var trackBuffers: [String: FrameBuffer] = [:]
    private var frameIndex = 0
    private var cancellables = Set<AnyCancellable>()

    var previewLayer: AVCaptureVideoPreviewLayer { camera.previewLayer }

    init(
        camera: CameraSessionProtocol = CameraSession(),
        estimator: any PoseEstimatorProtocol = PoseEstimator(),
        converter: any KeypointConverterProtocol = KeypointConverter(),
        scorer: any AnomalyScorerProtocol = AnomalyScorer(),
        tracking: TrackingServiceProtocol = TrackingService()
    ) {
        self.camera = camera
        self.estimator = estimator
        self.converter = converter
        self.scorer = scorer
        self.tracking = tracking
    }

    func enablePreviewTestMode() {
        detectionState = .warmingUp(framesCollected: 0, framesNeeded: FrameBuffer.capacity)
        skeletons = []
    }

    func start() throws {
        modelRunner = try? STGNFModelRunner()
        try camera.start()
        camera.framePublisher
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
        camera.stop()
        cancellables.removeAll()
        trackBuffers.removeAll()
        frameIndex = 0
        detectionState = .idle
    }

    // nonisolated so Vision and CoreML run on the cooperative thread pool,
    // never on the main thread. MainActor state is accessed via await MainActor.run.
    nonisolated private func processFrame(_ pixelBuffer: CVPixelBuffer) async {
        // Snapshot the Sendable objects we need off MainActor.
        let snapshot: (Int, any PoseEstimatorProtocol, any KeypointConverterProtocol, any AnomalyScorerProtocol, STGNFModelRunner?, UIDeviceOrientation)? = await MainActor.run { [weak self] in
            guard let self else { return nil }
            return (self.frameIndex, self.estimator, self.converter, self.scorer,
                    self.modelRunner, UIDevice.current.orientation)
        }
        guard let (currentFrameIndex, est, conv, scor, runner, deviceOrientation) = snapshot else { return }

        // Vision: runs on cooperative thread pool (NOT main thread).
        guard let observations = try? est.detectPoses(in: pixelBuffer,
                                                      deviceOrientation: deviceOrientation) else { return }

        let now = CMTime(seconds: Date().timeIntervalSince1970, preferredTimescale: 600)

        var currentSkeletons: [PoseSkeleton] = []

        for observation in observations {
            guard let skeleton = try? conv.convert(
                observation,
                frameIndex: currentFrameIndex,
                timestamp: now
            ) else { continue }
            currentSkeletons.append(skeleton)

            // matchTrack is MainActor-isolated — run on MainActor.
            let buffer: FrameBuffer = await MainActor.run { [weak self] in
                guard let self else { return FrameBuffer() }
                let trackID = tracking.matchTrack(for: skeleton)
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
                    let result = scor.classify(score: score, isWarmup: false)
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
}
