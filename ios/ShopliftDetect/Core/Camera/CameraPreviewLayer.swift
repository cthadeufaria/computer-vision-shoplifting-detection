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
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        hostedPreviewLayer.frame = bounds
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
    }
}
