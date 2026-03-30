import AVFoundation
import Combine

@MainActor
protocol CameraSessionProtocol: AnyObject {
    var framePublisher: AnyPublisher<CVPixelBuffer, Never> { get }
    var previewLayer: AVCaptureVideoPreviewLayer { get }
    func start() throws
    func stop()
}
