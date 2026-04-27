import SwiftUI

struct DeviceRowView: View {
    let tile: SupervisorFeedTileState
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            VStack(alignment: .leading, spacing: 8) {
                RoundedRectangle(cornerRadius: 16)
                    .fill(.thinMaterial)
                    .frame(height: 120)
                    .overlay(alignment: .center) {
                        VStack(spacing: 6) {
                            Image(systemName: tile.latestFrame == nil ? "video.slash" : "video")
                                .font(.system(size: 28))
                            Text(tile.statusText)
                                .font(.headline)
                        }
                    }
                    .overlay {
                        RoundedRectangle(cornerRadius: 16)
                            .stroke(tile.connectionState == .connected ? Color.green.opacity(0.45) : Color.secondary.opacity(0.35), lineWidth: 1)
                    }

                Text(tile.deviceName)
                    .font(.headline)

                if let anomalyBadgeText = tile.anomalyBadgeText {
                    Text(anomalyBadgeText)
                        .font(.caption.bold())
                        .foregroundStyle(anomalyBadgeText == "ANOMALY" ? .red : .secondary)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("supervisorTile_\(tile.deviceName)")
    }
}
