@preconcurrency import AVFoundation
import Combine
import CoreImage
import UIKit

@MainActor
final class CameraStreamingViewModel: ObservableObject {
    @Published private(set) var isStreaming = false
    @Published private(set) var statusText = "Ready to Stream"
    @Published private(set) var pairingPayloadText: String?

    private let camera: CameraSessionProtocol
    private let pairing: PairingServiceProtocol
    private let streaming: StreamingServiceProtocol
    private let browserStreaming: CameraFrameBroadcasting
    private let targetWidth: CGFloat
    private let jpegCompressionQuality: CGFloat
    private let browserFrameInterval: TimeInterval
    private var cancellables = Set<AnyCancellable>()
    private var isProcessingFrame = false
    private var lastBrowserFrameSentAt: Date?

    var previewLayer: AVCaptureVideoPreviewLayer { camera.previewLayer }

    init(
        camera: CameraSessionProtocol,
        pairing: PairingServiceProtocol,
        streaming: StreamingServiceProtocol,
        browserStreaming: CameraFrameBroadcasting = WebSocketCameraStreamingServer(),
        targetWidth: CGFloat = 640,
        jpegCompressionQuality: CGFloat = 0.55,
        browserFrameInterval: TimeInterval = 0.2
    ) {
        self.camera = camera
        self.pairing = pairing
        self.streaming = streaming
        self.browserStreaming = browserStreaming
        self.targetWidth = targetWidth
        self.jpegCompressionQuality = jpegCompressionQuality
        self.browserFrameInterval = browserFrameInterval
    }

    func start() throws {
        isProcessingFrame = false
        try camera.start()
        streaming.startStreaming()
        ensureCameraPairingPayload()
        try startBrowserStreamingIfPossible()
        isStreaming = true
        statusText = browserStreaming.isRunning
            ? "Streaming over local Wi-Fi to web supervisors"
            : "Streaming over local Wi-Fi"

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
        browserStreaming.stop()
        cancellables.removeAll()
        isStreaming = false
        isProcessingFrame = false
        lastBrowserFrameSentAt = nil
        pairingPayloadText = nil
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
            if shouldPublishBrowserFrame(now: Date()) {
                browserStreaming.publish(frame: frame)
                lastBrowserFrameSentAt = Date()
            }
        }
    }

    private func startBrowserStreamingIfPossible() throws {
        guard
            let session = pairing.currentSession,
            session.role == .camera,
            let token = pairing.currentToken,
            token.isVisibleOnScreen
        else {
            return
        }

        try browserStreaming.start(session: session, token: token)
    }

    private func ensureCameraPairingPayload() {
        if pairing.currentSession?.role == .camera,
           pairing.currentToken?.isVisibleOnScreen == true,
           let payload = pairing.qrPayloadString {
            pairingPayloadText = payload
            return
        }

        pairingPayloadText = pairing.prepareCameraPairing(deviceName: "Smart Camera")
    }

    private func shouldPublishBrowserFrame(now: Date) -> Bool {
        guard browserStreaming.isRunning else { return false }
        guard let lastBrowserFrameSentAt else { return true }
        return now.timeIntervalSince(lastBrowserFrameSentAt) >= browserFrameInterval
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
