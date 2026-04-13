import SwiftUI
import AVFoundation
import Combine
import CoreMedia
import UIKit

@MainActor
final class PosePreviewViewModel: ObservableObject {
    @Published var skeletons: [PoseSkeleton] = []
    @Published var debugInfo: String = ""

    private let camera: CameraSessionProtocol
    private let estimator: any PoseEstimatorProtocol
    private let converter: any KeypointConverterProtocol
    private var cancellables = Set<AnyCancellable>()
    private var isProcessingFrame = false
    private var frameIndex = 0
    private var processingSessionID = UUID()

    var previewLayer: AVCaptureVideoPreviewLayer { camera.previewLayer }

    init(
        camera: CameraSessionProtocol,
        estimator: any PoseEstimatorProtocol,
        converter: any KeypointConverterProtocol
    ) {
        self.camera = camera
        self.estimator = estimator
        self.converter = converter
    }

    func start() throws {
        processingSessionID = UUID()
        isProcessingFrame = false
        debugInfo = ""
        try camera.start()
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
    }

    func enablePreviewTestMode() {
        skeletons = [
            PoseSkeleton(
                keypoints: [],
                frameIndex: 0,
                timestamp: .zero,
                boundingBox: CGRect(x: 0.2, y: 0.2, width: 0.3, height: 0.4)
            )
        ]
        debugInfo = "UI Preview Mode"
    }

    func stop() {
        processingSessionID = UUID()
        camera.stop()
        cancellables.removeAll()
        isProcessingFrame = false
        frameIndex = 0
        skeletons = []
        debugInfo = ""
    }

    nonisolated private func processFrame(_ pixelBuffer: CVPixelBuffer, sessionID: UUID) async {
        let snapshot: (Int, any PoseEstimatorProtocol, any KeypointConverterProtocol, UIDeviceOrientation)? = await MainActor.run { [weak self] in
            guard let self else { return nil }
            guard self.processingSessionID == sessionID else { return nil }
            return (self.frameIndex, self.estimator, self.converter,
                    UIDevice.current.orientation)
        }
        guard let (currentFrameIndex, est, conv, deviceOrientation) = snapshot else { return }

        // Capture buffer dimensions before detection — tells us which orientation branch is taken.
        let bufW = CVPixelBufferGetWidth(pixelBuffer)
        let bufH = CVPixelBufferGetHeight(pixelBuffer)
        let chosenOrientation = PoseEstimator.imageOrientation(for: pixelBuffer,
                                                               deviceOrientation: deviceOrientation)

        guard let observations = try? est.detectPoses(in: pixelBuffer,
                                                      deviceOrientation: deviceOrientation) else { return }

        let now = CMTime(seconds: Date().timeIntervalSince1970, preferredTimescale: 600)

        let currentSkeletons = observations.compactMap { observation in
            try? conv.convert(
                observation,
                frameIndex: currentFrameIndex,
                timestamp: now
            )
        }

        // Extract all debug values before crossing to MainActor (only plain value types cross the boundary).
        let personCount = observations.count
        var rawNose: CGPoint? = nil
        var flippedNose: CGPoint? = nil
        // Keypoint index 0 is nose in COCO18 order (KeypointConverter oppOrder[0] = 0).
        // Coordinates are UIKit-normalized (y-flipped from Vision), so invert y to recover Vision coords.
        if let kp = currentSkeletons.first?.keypoints.first, kp.confidence > 0.1 {
            flippedNose = CGPoint(x: CGFloat(kp.x), y: CGFloat(kp.y))
            rawNose     = CGPoint(x: CGFloat(kp.x), y: 1 - CGFloat(kp.y))
        }

        await MainActor.run { [weak self] in
            guard let self else { return }
            guard self.processingSessionID == sessionID else { return }
            skeletons = currentSkeletons
            frameIndex += 1

            let orientLabel: String
            switch chosenOrientation {
            case .up:    orientLabel = ".up (portrait, no rotation)"
            case .down:  orientLabel = ".down (portrait upside-down)"
            case .right: orientLabel = ".right (landscape→portrait, 90°CCW)"
            case .left:  orientLabel = ".left (landscape→portrait, 90°CW)"
            default:     orientLabel = "other(\(chosenOrientation.rawValue))"
            }

            var lines = [
                "Buffer: \(bufW)×\(bufH) (\(bufH > bufW ? "portrait" : "landscape"))",
                "DeviceOrient: \(deviceOrientation.debugName)",
                "VisionOrient: \(orientLabel)",
                "Persons: \(personCount)",
            ]
            if let rn = rawNose, let fn = flippedNose {
                lines.append(String(format: "Nose raw (Vision): (%.3f, %.3f)", rn.x, rn.y))
                lines.append(String(format: "Nose flipped (UIKit): (%.3f, %.3f)", fn.x, fn.y))
            } else {
                lines.append("Nose: not detected")
            }
            debugInfo = lines.joined(separator: "\n")
        }
    }
}

private extension UIDeviceOrientation {
    var debugName: String {
        switch self {
        case .portrait:            return "portrait"
        case .portraitUpsideDown:  return "portraitUpsideDown"
        case .landscapeLeft:       return "landscapeLeft"
        case .landscapeRight:      return "landscapeRight"
        case .faceUp:              return "faceUp"
        case .faceDown:            return "faceDown"
        default:                   return "unknown"
        }
    }
}
