import SwiftUI
import AVFoundation

/// Draws skeleton bones over the camera preview using Canvas.
struct SkeletonOverlayView: View {
    let skeletons: [PoseSkeleton]
    /// The same preview layer that displays the camera feed.
    /// Used to convert Vision's normalised keypoint coordinates — which are in the
    /// capture-device space of the *full* uncropped frame — into screen points that
    /// account for the layer's videoGravity (resizeAspectFill cropping).
    let previewLayer: AVCaptureVideoPreviewLayer

    // COCO18 bone pairs (indices into the 18-keypoint array).
    private static let bones: [(Int, Int)] = [
        (0, 1), (1, 2), (2, 3), (3, 4),       // head chain
        (1, 5), (5, 6), (6, 7),                // left arm
        (1, 8), (8, 9), (9, 10),               // right arm
        (1, 11), (11, 12), (12, 13),           // left leg
        (1, 14), (14, 15), (15, 16),           // right leg
        (11, 14)                               // hip cross
    ]

    var body: some View {
        Canvas { context, size in
            // Convert a keypoint from capture-device normalised space (0–1, top-left origin,
            // matching what KeypointConverter produces) into Canvas points.
            // layerPointConverted handles videoGravity (aspect-fill crop) and any rotation.
            // We then re-normalise by the layer bounds so the result scales with the canvas.
            func pt(_ kp: Keypoint) -> CGPoint {
                let lw = previewLayer.bounds.width
                let lh = previewLayer.bounds.height
                guard lw > 0, lh > 0 else {
                    return CGPoint(x: CGFloat(kp.x) * size.width, y: CGFloat(kp.y) * size.height)
                }
                let layerPt = previewLayer.layerPointConverted(
                    fromCaptureDevicePoint: CGPoint(x: Double(kp.x), y: Double(kp.y))
                )
                return CGPoint(x: layerPt.x / lw * size.width, y: layerPt.y / lh * size.height)
            }

            for skeleton in skeletons {
                let kps = skeleton.keypoints
                for (a, b) in Self.bones {
                    guard a < kps.count, b < kps.count,
                          kps[a].confidence > 0.3, kps[b].confidence > 0.3 else { continue }
                    let path = Path { p in
                        p.move(to: pt(kps[a]))
                        p.addLine(to: pt(kps[b]))
                    }
                    context.stroke(path, with: .color(.green.opacity(0.8)), lineWidth: 2)
                }
                for kp in kps where kp.confidence > 0.3 {
                    let c = pt(kp)
                    let rect = CGRect(x: c.x - 3, y: c.y - 3, width: 6, height: 6)
                    context.fill(Path(ellipseIn: rect), with: .color(.yellow))
                }
            }
        }
        .allowsHitTesting(false)
    }
}
