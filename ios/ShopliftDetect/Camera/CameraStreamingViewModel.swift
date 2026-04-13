@preconcurrency import AVFoundation
import Combine
import CoreImage
import UIKit

@MainActor
final class CameraStreamingViewModel: ObservableObject {
    @Published private(set) var isStreaming = false
    @Published private(set) var statusText = "Ready to Stream"

    private let camera: CameraSessionProtocol
    private let pairing: PairingServiceProtocol
    private let streaming: StreamingServiceProtocol
    private let targetWidth: CGFloat
    private let jpegCompressionQuality: CGFloat
    private var cancellables = Set<AnyCancellable>()
    private var isProcessingFrame = false

    var previewLayer: AVCaptureVideoPreviewLayer { camera.previewLayer }

    init(
        camera: CameraSessionProtocol,
        pairing: PairingServiceProtocol,
        streaming: StreamingServiceProtocol,
        targetWidth: CGFloat = 640,
        jpegCompressionQuality: CGFloat = 0.55
    ) {
        self.camera = camera
        self.pairing = pairing
        self.streaming = streaming
        self.targetWidth = targetWidth
        self.jpegCompressionQuality = jpegCompressionQuality
    }

    func start() throws {
        isProcessingFrame = false
        try camera.start()
        streaming.startStreaming()
        isStreaming = true
        statusText = "Streaming over local Wi-Fi"

        camera.framePublisher
            .sink { [weak self] pixelBuffer in
                guard let self else { return }
                guard !isProcessingFrame else { return }
                isProcessingFrame = true
                Task { [weak self] in
                    guard let self else { return }
                    await self.publish(pixelBuffer: pixelBuffer)
                    await MainActor.run {
                        self.isProcessingFrame = false
                    }
                }
            }
            .store(in: &cancellables)
    }

    func stop() {
        camera.stop()
        streaming.stopStreaming()
        cancellables.removeAll()
        isStreaming = false
        isProcessingFrame = false
        statusText = "Streaming Stopped"
    }

    nonisolated private func publish(pixelBuffer: CVPixelBuffer) async {
        let configuration = await MainActor.run { [targetWidth, jpegCompressionQuality] in
            (targetWidth, jpegCompressionQuality)
        }
        guard let frame = Self.makeVideoFrame(
            from: pixelBuffer,
            targetWidth: configuration.0,
            jpegCompressionQuality: configuration.1
        ) else { return }
        await MainActor.run { [weak self] in
            guard let self else { return }
            guard let sessionID = pairing.currentSession?.sessionID else { return }
            streaming.publishFrame(frame, for: sessionID)
        }
    }

    nonisolated private static func makeVideoFrame(
        from pixelBuffer: CVPixelBuffer,
        targetWidth: CGFloat,
        jpegCompressionQuality: CGFloat
    ) -> VideoFrame? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let ciContext = CIContext()
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else {
            return nil
        }

        let image = UIImage(cgImage: cgImage)
        let scale = min(1, targetWidth / max(image.size.width, 1))
        let targetSize = CGSize(
            width: max(1, image.size.width * scale),
            height: max(1, image.size.height * scale)
        )
        let renderer = UIGraphicsImageRenderer(size: targetSize)
        let rendered = renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: targetSize))
        }

        guard let jpegData = rendered.jpegData(compressionQuality: jpegCompressionQuality) else {
            return nil
        }

        return VideoFrame(
            timestamp: UInt64(Date().timeIntervalSince1970 * 1000),
            jpegData: jpegData,
            width: Int(targetSize.width),
            height: Int(targetSize.height)
        )
    }
}
