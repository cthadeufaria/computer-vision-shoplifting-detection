import SwiftUI
import UIKit
import AVFoundation

struct CameraPreviewLayer: UIViewRepresentable {
    let previewLayer: AVCaptureVideoPreviewLayer

    func makeUIView(context: Context) -> UIView {
        PreviewContainerView(previewLayer: previewLayer)
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        guard let view = uiView as? PreviewContainerView else { return }
        view.setPreviewLayer(previewLayer)
    }
}

private final class PreviewContainerView: UIView {
    private var hostedPreviewLayer: AVCaptureVideoPreviewLayer

    init(previewLayer: AVCaptureVideoPreviewLayer) {
        hostedPreviewLayer = previewLayer
        super.init(frame: .zero)
        isAccessibilityElement = true
        accessibilityIdentifier = "cameraPreview"
        backgroundColor = .black
        attachPreviewLayer()
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(orientationDidChange),
            name: UIDevice.orientationDidChangeNotification,
            object: nil
        )
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        hostedPreviewLayer.frame = bounds
        updatePreviewRotation()
    }

    func setPreviewLayer(_ previewLayer: AVCaptureVideoPreviewLayer) {
        guard hostedPreviewLayer !== previewLayer else {
            hostedPreviewLayer.frame = bounds
            return
        }
        hostedPreviewLayer.removeFromSuperlayer()
        hostedPreviewLayer = previewLayer
        attachPreviewLayer()
    }

    private func attachPreviewLayer() {
        hostedPreviewLayer.removeFromSuperlayer()
        layer.addSublayer(hostedPreviewLayer)
        hostedPreviewLayer.frame = bounds
        updatePreviewRotation()
    }

    @objc private func orientationDidChange() {
        updatePreviewRotation()
    }

    private func updatePreviewRotation() {
        let angle: CGFloat
        switch UIDevice.current.orientation {
        case .landscapeLeft:      angle = 0    // top of device on left  → natural landscape
        case .landscapeRight:     angle = 180  // top of device on right → inverted landscape
        case .portraitUpsideDown: angle = 270
        default:                  angle = 90   // portrait and face-up/down: keep portrait
        }
        guard let connection = hostedPreviewLayer.connection,
              connection.isVideoRotationAngleSupported(angle) else { return }
        connection.videoRotationAngle = angle
    }
}
