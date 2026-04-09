import SwiftUI

struct DeviceRowView: View {
    let tile: SupervisorFeedTileState
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            VStack(alignment: .leading, spacing: 8) {
                RoundedRectangle(cornerRadius: 16)
                    .fill(tile.connectionState == .connected ? Color.green.opacity(0.18) : Color.gray.opacity(0.18))
                    .frame(height: 120)
                    .overlay(alignment: .center) {
                        VStack(spacing: 6) {
                            Image(systemName: tile.latestFrame == nil ? "video.slash" : "video")
                                .font(.system(size: 28))
                            Text(tile.statusText)
                                .font(.headline)
                        }
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

