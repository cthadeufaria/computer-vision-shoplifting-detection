import SwiftUI
import AVFoundation
import Combine
import CoreMedia

@MainActor
final class DetectionViewModel: ObservableObject {
    @Published var detectionState: DetectionState = .idle
    @Published var skeletons: [PoseSkeleton] = []

    private let cameraSession = CameraSession()
    private let poseEstimator = PoseEstimator()
    private let keypointConverter = KeypointConverter()
    private let anomalyScorer = AnomalyScorer()

    // Per-track frame buffers keyed by IoU-matched bounding box track IDs.
    private var trackBuffers: [String: FrameBuffer] = [:]
    private var frameIndex = 0
    private var cancellables = Set<AnyCancellable>()

    var previewLayer: AVCaptureVideoPreviewLayer { cameraSession.previewLayer }

    func start() throws {
        try cameraSession.start()
        cameraSession.framePublisher
            .receive(on: DispatchQueue.global(qos: .userInitiated))
            .sink { [weak self] pixelBuffer in
                Task { @MainActor [weak self] in
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

    private func processFrame(_ pixelBuffer: CVPixelBuffer) async {
        guard let observations = try? poseEstimator.detectPoses(in: pixelBuffer) else { return }

        let width = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let height = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        let previewSize = CGSize(width: width, height: height)
        let now = CMTime(seconds: Date().timeIntervalSince1970, preferredTimescale: 600)

        var currentSkeletons: [PoseSkeleton] = []
        for observation in observations {
            guard let skeleton = try? keypointConverter.convert(
                observation, previewSize: previewSize,
                frameIndex: frameIndex, timestamp: now
            ) else { continue }
            currentSkeletons.append(skeleton)

            let trackID = matchTrack(for: skeleton)
            let buffer = trackBuffers[trackID, default: FrameBuffer()]
            trackBuffers[trackID] = buffer
            await buffer.append(skeleton)

            if await buffer.isReady, let window = await buffer.currentWindow() {
                let normalizer = PoseNormalizer(videoWidth: Float(width), videoHeight: Float(height))
                if let mlArray = try? normalizer.normalize(window),
                   let score = try? STGNFModelRunner().runInference(on: mlArray) {
                    let result = anomalyScorer.classify(score: score, isWarmup: false)
                    detectionState = .running(latestResult: result)
                }
            }
        }

        skeletons = currentSkeletons
        frameIndex += 1

        // Update warmup state from the first tracked person's buffer.
        if case .warmingUp = detectionState, let firstBuffer = trackBuffers.values.first {
            let count = await firstBuffer.count
            detectionState = .warmingUp(framesCollected: count, framesNeeded: FrameBuffer.capacity)
        }
    }

    // Simple IoU-based tracker: find the best matching existing track or create a new one.
    private func matchTrack(for skeleton: PoseSkeleton) -> String {
        let iouThreshold: CGFloat = 0.3
        var bestID: String?
        var bestIoU: CGFloat = 0

        for (id, _) in trackBuffers {
            // We store last known bbox in a parallel dict; for simplicity use first skeleton match.
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
