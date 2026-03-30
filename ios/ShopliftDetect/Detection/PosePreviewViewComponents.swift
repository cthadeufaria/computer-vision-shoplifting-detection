import SwiftUI

struct PosePreviewTopBar: View {
    let skeletonCount: Int
    let rotation: Angle
    let onDismiss: () -> Void

    var body: some View {
        VStack {
            HStack {
                Button(action: onDismiss) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title)
                        .foregroundStyle(.white)
                }
                .accessibilityIdentifier("posePreviewDismissButton")
                .rotationEffect(rotation)
                .padding()

                Spacer()

                Text("Poses: \(skeletonCount)")
                    .font(.caption.bold())
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(.ultraThinMaterial)
                    .clipShape(Capsule())
                    .accessibilityIdentifier("posePreviewCount")
                    .rotationEffect(rotation)
                    .padding()
            }
            Spacer()
        }
    }
}

struct PoseDebugOverlay: View {
    let debugInfo: String

    var body: some View {
        if !debugInfo.isEmpty {
            VStack {
                Spacer()
                Text(debugInfo)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(.white)
                    .padding(8)
                    .background(Color.black.opacity(0.65))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 12)
                    .padding(.bottom, 12)
            }
        }
    }
}
