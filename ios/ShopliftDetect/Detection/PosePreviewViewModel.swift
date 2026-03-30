import SwiftUI
import AVFoundation
import Combine
import CoreMedia
import UIKit

@MainActor
final class PosePreviewViewModel: ObservableObject {
    @Published var skeletons: [PoseSkeleton] = []

    private let camera: CameraSessionProtocol
    private let estimator: any PoseEstimatorProtocol
    private let converter: any KeypointConverterProtocol
    private var cancellables = Set<AnyCancellable>()
    private var frameIndex = 0

    var previewLayer: AVCaptureVideoPreviewLayer { camera.previewLayer }

    init(
        camera: CameraSessionProtocol = CameraSession(),
        estimator: any PoseEstimatorProtocol = PoseEstimator(),
        converter: any KeypointConverterProtocol = KeypointConverter()
    ) {
        self.camera = camera
        self.estimator = estimator
        self.converter = converter
    }

    func start() throws {
        try camera.start()
        camera.framePublisher
            .sink { [weak self] pixelBuffer in
                guard let self else { return }
                Task { [weak self] in
                    await self?.processFrame(pixelBuffer)
                }
            }
            .store(in: &cancellables)
    }

    func stop() {
        camera.stop()
        cancellables.removeAll()
        frameIndex = 0
        skeletons = []
    }

    nonisolated private func processFrame(_ pixelBuffer: CVPixelBuffer) async {
        let snapshot: (Int, any PoseEstimatorProtocol, any KeypointConverterProtocol, UIDeviceOrientation)? = await MainActor.run { [weak self] in
            guard let self else { return nil }
            return (self.frameIndex, self.estimator, self.converter,
                    UIDevice.current.orientation)
        }
        guard let (currentFrameIndex, est, conv, deviceOrientation) = snapshot else { return }
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

        await MainActor.run { [weak self] in
            guard let self else { return }
            skeletons = currentSkeletons
            frameIndex += 1
        }
    }
}
