import SwiftUI

struct CameraFeedDetailView: View {
    let tile: SupervisorFeedTileState
    let onDismiss: () -> Void

    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                RoundedRectangle(cornerRadius: 20)
                    .fill(.thinMaterial)
                    .frame(height: 260)
                    .overlay {
                        VStack(spacing: 10) {
                            Image(systemName: tile.latestFrame == nil ? "video.slash" : "video.fill")
                                .font(.system(size: 40))
                            Text(tile.statusText)
                                .font(.title3.bold())
                        }
                    }
                    .overlay {
                        RoundedRectangle(cornerRadius: 20)
                            .stroke(tile.connectionState == .connected ? Color.green.opacity(0.45) : Color.secondary.opacity(0.35), lineWidth: 1)
                    }

                Text(tile.deviceName)
                    .font(.title2.bold())
                    .accessibilityIdentifier("cameraFeedDetailTitle")

                if let anomalyBadgeText = tile.anomalyBadgeText {
                    Text(anomalyBadgeText)
                        .font(.headline)
                }

                Spacer()
            }
            .padding()
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done", action: onDismiss)
                }
            }
        }
        .navigationViewStyle(.stack)
        .screenAppearanceIdentifier("cameraFeedDetailScreen")
    }
}
