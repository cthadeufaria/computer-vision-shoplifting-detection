import SwiftUI

/// Draws skeleton bones over the camera preview using Canvas.
struct SkeletonOverlayView: View {
    let skeletons: [PoseSkeleton]

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
        // Canvas provides its own size in points. Keypoints are in normalized (0–1) space,
        // so multiply by size to get the correct screen position regardless of device.
        Canvas { context, size in
            for skeleton in skeletons {
                let kps = skeleton.keypoints
                for (a, b) in Self.bones {
                    guard a < kps.count, b < kps.count,
                          kps[a].confidence > 0.3, kps[b].confidence > 0.3 else { continue }
                    let path = Path { p in
                        p.move(to: CGPoint(
                            x: CGFloat(kps[a].x) * size.width,
                            y: CGFloat(kps[a].y) * size.height
                        ))
                        p.addLine(to: CGPoint(
                            x: CGFloat(kps[b].x) * size.width,
                            y: CGFloat(kps[b].y) * size.height
                        ))
                    }
                    context.stroke(path, with: .color(.green.opacity(0.8)), lineWidth: 2)
                }
                for kp in kps where kp.confidence > 0.3 {
                    let cx = CGFloat(kp.x) * size.width
                    let cy = CGFloat(kp.y) * size.height
                    let rect = CGRect(x: cx - 3, y: cy - 3, width: 6, height: 6)
                    context.fill(Path(ellipseIn: rect), with: .color(.yellow))
                }
            }
        }
        .allowsHitTesting(false)
    }
}
