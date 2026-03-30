@preconcurrency import AVFoundation
import Combine
import UIKit

/// Manages AVCaptureSession and publishes CVPixelBuffer frames.
@MainActor
final class CameraSession: NSObject, ObservableObject, CameraSessionProtocol {
    private let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "com.shopliftdetect.camera", qos: .userInitiated)
    let previewLayer: AVCaptureVideoPreviewLayer

    private let frameSubject = PassthroughSubject<CVPixelBuffer, Never>()
    var framePublisher: AnyPublisher<CVPixelBuffer, Never> { frameSubject.eraseToAnyPublisher() }
    
    override init() {
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.videoGravity = .resizeAspectFill
        super.init()
    }

    func start() throws {
        guard AVCaptureDevice.authorizationStatus(for: .video) == .authorized else {
            throw CameraError.permissionDenied
        }

        captureSession.beginConfiguration()
        captureSession.sessionPreset = .hd1920x1080

        // Prefer back camera on iPhone; fall back to any available camera (Mac/simulator).
        if captureSession.inputs.isEmpty {
            let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
                ?? AVCaptureDevice.default(for: .video)
            guard let device,
                  let input = try? AVCaptureDeviceInput(device: device),
                  captureSession.canAddInput(input) else {
                throw CameraError.deviceUnavailable
            }
            captureSession.addInput(input)
        }

        if captureSession.outputs.isEmpty {
            videoOutput.setSampleBufferDelegate(self, queue: sessionQueue)
            videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            guard captureSession.canAddOutput(videoOutput) else {
                throw CameraError.outputUnavailable
            }
            captureSession.addOutput(videoOutput)

            // Rotate buffer to portrait so Vision receives an upright image and can use
            // .up orientation (no rotation math). This also aligns layerPointConverted's
            // coordinate space with Vision's output, so skeleton overlay requires no inversion.
            if let connection = videoOutput.connection(with: .video),
               connection.isVideoRotationAngleSupported(90) {
                connection.videoRotationAngle = 90
            }
        }
        captureSession.commitConfiguration()

        sessionQueue.async { [captureSession] in
            captureSession.startRunning()
        }
        // Required for UIDevice.current.orientation to reflect device rotation.
        UIDevice.current.beginGeneratingDeviceOrientationNotifications()
    }

    func stop() {
        UIDevice.current.endGeneratingDeviceOrientationNotifications()
        sessionQueue.async { [captureSession] in
            captureSession.stopRunning()
        }
    }
}

extension CameraSession: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        Task { @MainActor in
            self.frameSubject.send(pixelBuffer)
        }
    }
}

enum CameraError: Error, LocalizedError {
    case permissionDenied
    case deviceUnavailable
    case outputUnavailable

    var errorDescription: String? {
        switch self {
        case .permissionDenied: "Camera access is required. Enable it in Settings → ShopliftDetect → Camera."
        case .deviceUnavailable: "No camera device found."
        case .outputUnavailable: "Could not configure camera output."
        }
    }
}
