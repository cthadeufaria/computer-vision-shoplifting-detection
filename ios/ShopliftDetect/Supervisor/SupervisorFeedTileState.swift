import Foundation

struct SupervisorFeedTileState: Equatable, Sendable, Identifiable {
    let sessionID: UUID
    let deviceName: String
    var connectionState: ConnectionState
    var latestFrame: VideoFrame?
    var latestDetections: [DetectionResult]

    var id: UUID { sessionID }

    var statusText: String {
        switch connectionState {
        case .connected:
            return "Connected"
        case .stale:
            return "Stale"
        case .disconnected:
            return "Disconnected"
        case .failed:
            return "Failed"
        case .connecting, .handshaking:
            return "Connecting"
        case .listening:
            return "Listening"
        case .idle:
            return "Idle"
        }
    }

    var anomalyBadgeText: String? {
        guard let latest = latestDetections.last else { return nil }
        switch latest.label {
        case .anomaly:
            return "ANOMALY"
        case .warmup:
            return "WARMING UP"
        case .normal:
            return "GOOD"
        }
    }
}

