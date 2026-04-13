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
    private var scorer: any AnomalyScorerProtocol
    private let tracking: TrackingServiceProtocol
    private let settings: SettingsServiceProtocol
    private let streaming: StreamingServiceProtocol
    private var modelRunner: STGNFModelRunner?

    // Per-track frame buffers keyed by IoU-matched bounding box track IDs.
    private var trackBuffers: [String: FrameBuffer] = [:]
    private var isProcessingFrame = false
    private var frameIndex = 0
    private var cancellables = Set<AnyCancellable>()
    private var processingSessionID = UUID()

    var previewLayer: AVCaptureVideoPreviewLayer { camera.previewLayer }

    init(
        camera: CameraSessionProtocol,
        estimator: any PoseEstimatorProtocol,
        converter: any KeypointConverterProtocol,
        scorer: any AnomalyScorerProtocol,
        tracking: TrackingServiceProtocol,
        settings: SettingsServiceProtocol,
        streaming: StreamingServiceProtocol
    ) {
        self.camera = camera
        self.estimator = estimator
        self.converter = converter
        self.scorer = scorer
        self.tracking = tracking
        self.settings = settings
        self.streaming = streaming
    }

    var threshold: Float {
        settings.anomalyThreshold
    }

    var isStreaming: Bool {
        streaming.isStreaming
    }

    func enablePreviewTestMode() {
        detectionState = .warmingUp(framesCollected: 0, framesNeeded: FrameBuffer.capacity)
        skeletons = []
    }

    func updateThreshold(_ newValue: Float) {
        settings.anomalyThreshold = newValue
        scorer.threshold = newValue
        objectWillChange.send()
    }

    func start() throws {
        modelRunner = try? STGNFModelRunner()
        processingSessionID = UUID()
        isProcessingFrame = false
        try camera.start()
        streaming.startStreaming()
        let sessionID = processingSessionID
        camera.framePublisher
            .sink { [weak self] pixelBuffer in
                guard let self else { return }
                guard !isProcessingFrame else { return }
                isProcessingFrame = true
                Task { [weak self] in
                    guard let self else { return }
                    await self.processFrame(pixelBuffer, sessionID: sessionID)
                    await MainActor.run {
                        guard self.processingSessionID == sessionID else { return }
                        self.isProcessingFrame = false
                    }
                }
            }
            .store(in: &cancellables)
        detectionState = .warmingUp(framesCollected: 0, framesNeeded: FrameBuffer.capacity)
    }

    func stop() {
        processingSessionID = UUID()
        camera.stop()
        streaming.stopStreaming()
        for feed in streaming.feedStates {
            streaming.updateFeedConnectionState(.stale, for: feed.sessionID)
        }
        cancellables.removeAll()
        trackBuffers.removeAll()
        isProcessingFrame = false
        frameIndex = 0
        detectionState = .idle
        skeletons = []
    }

    // nonisolated so Vision and CoreML run on the cooperative thread pool,
    // never on the main thread. MainActor state is accessed via await MainActor.run.
    nonisolated private func processFrame(_ pixelBuffer: CVPixelBuffer, sessionID: UUID) async {
        // Snapshot the Sendable objects we need off MainActor.
        let snapshot: (Int, any PoseEstimatorProtocol, any KeypointConverterProtocol, any AnomalyScorerProtocol, STGNFModelRunner?, UIDeviceOrientation)? = await MainActor.run { [weak self] in
            guard let self else { return nil }
            guard self.processingSessionID == sessionID else { return nil }
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
                        guard let self else { return }
                        guard self.processingSessionID == sessionID else { return }
                        self.detectionState = .running(latestResult: result)
                    }
                }
            }
        }

        // Commit UI updates.
        await MainActor.run { [weak self] in
            guard let self else { return }
            guard self.processingSessionID == sessionID else { return }
            skeletons = currentSkeletons
            frameIndex += 1
            publishSupervisorUpdates(for: currentSkeletons)
        }

        // Warmup counter update (FrameBuffer.count is async — actor isolated).
        let firstBuffer: FrameBuffer? = await MainActor.run { [weak self] in
            guard let self else { return nil }
            guard self.processingSessionID == sessionID else { return nil }
            return self.trackBuffers.values.first
        }
        if let firstBuffer {
            let count = await firstBuffer.count
            await MainActor.run { [weak self] in
                guard let self, case .warmingUp = detectionState else { return }
                guard self.processingSessionID == sessionID else { return }
                detectionState = .warmingUp(framesCollected: count, framesNeeded: FrameBuffer.capacity)
            }
        }
    }

    private func publishSupervisorUpdates(for skeletons: [PoseSkeleton]) {
        guard let sessionID = streaming.feedStates.first?.sessionID else { return }

        let detections = skeletons.enumerated().map { index, skeleton in
            DetectionResult(
                trackID: index + 1,
                score: scoreForCurrentState(),
                label: labelForCurrentState(),
                keypoints: skeleton.keypoints,
                boundingBox: skeleton.boundingBox,
                timestamp: Date()
            )
        }

        let frame = VideoFrame(
            timestamp: UInt64(Date().timeIntervalSince1970 * 1000),
            jpegData: Data([UInt8(min(skeletons.count, 255))]),
            width: 320,
            height: 240
        )

        streaming.publishFrame(frame, for: sessionID)
        streaming.publishDetections(detections, for: sessionID)
    }

    private func scoreForCurrentState() -> Float {
        switch detectionState {
        case .running(let latestResult):
            return latestResult.score
        case .warmingUp:
            return 0
        case .error:
            return -999
        case .idle:
            return 0
        }
    }

    private func labelForCurrentState() -> AnomalyLabel {
        switch detectionState {
        case .running(let latestResult):
            return latestResult.label
        case .warmingUp:
            return .warmup
        case .error:
            return .anomaly
        case .idle:
            return .normal
        }
    }
}
