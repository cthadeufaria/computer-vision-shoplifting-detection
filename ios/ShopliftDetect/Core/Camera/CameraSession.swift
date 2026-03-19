@preconcurrency import AVFoundation
import Combine

/// Manages AVCaptureSession and publishes CVPixelBuffer frames.
@MainActor
final class CameraSession: NSObject, ObservableObject {
    private let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "com.shopliftdetect.camera", qos: .userInitiated)

    private let frameSubject = PassthroughSubject<CVPixelBuffer, Never>()
    var framePublisher: AnyPublisher<CVPixelBuffer, Never> { frameSubject.eraseToAnyPublisher() }

    var previewLayer: AVCaptureVideoPreviewLayer {
        AVCaptureVideoPreviewLayer(session: captureSession)
    }

    func start() throws {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .hd1920x1080

        // Prefer back camera on iPhone; fall back to any available camera (Mac/simulator).
        let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
            ?? AVCaptureDevice.default(for: .video)
        guard let device,
              let input = try? AVCaptureDeviceInput(device: device),
              captureSession.canAddInput(input) else {
            throw CameraError.deviceUnavailable
        }
        captureSession.addInput(input)

        videoOutput.setSampleBufferDelegate(self, queue: sessionQueue)
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        guard captureSession.canAddOutput(videoOutput) else {
            throw CameraError.outputUnavailable
        }
        captureSession.addOutput(videoOutput)
        captureSession.commitConfiguration()

        sessionQueue.async { [captureSession] in
            captureSession.startRunning()
        }
    }

    func stop() {
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

enum CameraError: Error {
    case deviceUnavailable
    case outputUnavailable
}
