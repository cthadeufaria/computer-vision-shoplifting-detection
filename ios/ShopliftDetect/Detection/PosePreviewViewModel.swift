import SwiftUI
import AVFoundation
import Combine
import CoreMedia
import UIKit

@MainActor
final class PosePreviewViewModel: ObservableObject {
    @Published var skeletons: [PoseSkeleton] = []

    private let cameraSession = CameraSession()
    private let poseEstimator = PoseEstimator()
    private let keypointConverter = KeypointConverter()
    private var cancellables = Set<AnyCancellable>()
    private var frameIndex = 0

    var previewLayer: AVCaptureVideoPreviewLayer { cameraSession.previewLayer }

    func start() throws {
        try cameraSession.start()
        cameraSession.framePublisher
            .sink { [weak self] pixelBuffer in
                guard let self else { return }
                Task { [weak self] in
                    await self?.processFrame(pixelBuffer)
                }
            }
            .store(in: &cancellables)
    }

    func stop() {
        cameraSession.stop()
        cancellables.removeAll()
        frameIndex = 0
        skeletons = []
    }

    nonisolated private func processFrame(_ pixelBuffer: CVPixelBuffer) async {
        let snapshot: (Int, PoseEstimator, KeypointConverter, UIDeviceOrientation)? = await MainActor.run { [weak self] in
            guard let self else { return nil }
            return (self.frameIndex, self.poseEstimator, self.keypointConverter,
                    UIDevice.current.orientation)
        }
        guard let (currentFrameIndex, estimator, converter, deviceOrientation) = snapshot else { return }
        guard let observations = try? estimator.detectPoses(in: pixelBuffer,
                                                            deviceOrientation: deviceOrientation) else { return }

        let now = CMTime(seconds: Date().timeIntervalSince1970, preferredTimescale: 600)

        let currentSkeletons = observations.compactMap { observation in
            try? converter.convert(
                observation,
                frameIndex: currentFrameIndex,
                timestamp: now
            )
        }

        await MainActor.run { [weak self] in
            guard let self else { return }
            skeletons = currentSkeletons
            frameIndex += 1
        }
    }
}
