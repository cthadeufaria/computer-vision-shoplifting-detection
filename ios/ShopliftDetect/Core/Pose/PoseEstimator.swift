import Vision
import CoreImage
import UIKit

/// Runs VNDetectHumanBodyPoseRequest on camera frames.
final class PoseEstimator: @unchecked Sendable {
    private let requestHandler = VNSequenceRequestHandler()

    func detectPoses(
        in pixelBuffer: CVPixelBuffer,
        deviceOrientation: UIDeviceOrientation = .portrait
    ) throws -> [VNHumanBodyPoseObservation] {
        let request = VNDetectHumanBodyPoseRequest()
        try requestHandler.perform([request], on: pixelBuffer,
                                   orientation: Self.imageOrientation(for: pixelBuffer,
                                                                      deviceOrientation: deviceOrientation))
        return request.results ?? []
    }

    /// Returns the CGImagePropertyOrientation Vision needs for the given pixel buffer.
    ///
    /// On iOS 17+ AVCaptureVideoDataOutput delivers portrait-oriented buffers by default
    /// (height > width). On older iOS the buffer is always landscape (width > height).
    /// We check the actual dimensions rather than assuming, so the same code works on
    /// both generations.
    ///
    /// Portrait buffer (h > w) — no rotation needed, just handle upside-down:
    ///   portrait / default  → .up
    ///   portraitUpsideDown  → .down
    ///
    /// Landscape buffer (w > h) — camera sensor native; device orientation drives correction:
    ///   portrait            → .right  (rotate 90° CCW)
    ///   portraitUpsideDown  → .left   (rotate 90° CW)
    ///   landscapeRight      → .up     (no rotation — native sensor orientation)
    ///   landscapeLeft       → .down   (rotate 180°)
    static func imageOrientation(
        for pixelBuffer: CVPixelBuffer,
        deviceOrientation: UIDeviceOrientation
    ) -> CGImagePropertyOrientation {
        let w = CVPixelBufferGetWidth(pixelBuffer)
        let h = CVPixelBufferGetHeight(pixelBuffer)

        if h > w {
            // Buffer is already portrait-oriented.
            return deviceOrientation == .portraitUpsideDown ? .down : .up
        } else {
            // Landscape buffer — rotate to match the device's current orientation.
            switch deviceOrientation {
            case .portrait:           return .right
            case .portraitUpsideDown: return .left
            case .landscapeRight:     return .up
            case .landscapeLeft:      return .down
            default:                  return .right
            }
        }
    }
}
