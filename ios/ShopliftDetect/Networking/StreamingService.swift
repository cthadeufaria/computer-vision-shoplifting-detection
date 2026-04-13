import Foundation

@MainActor
final class StreamingService: StreamingServiceProtocol {
    private(set) var isStreaming = false
    private(set) var connectionState: ConnectionState = .idle
    private(set) var latestHeartbeatAt: Date?
    private(set) var feedStates: [SupervisorFeedTileState] = []
    private(set) var isEncryptedChannelEstablished = false

    private let streamProtocol: StreamProtocol
    private let nowProvider: () -> Date
    private let timeoutInterval: TimeInterval
    private let secureTransport: any SecureTransportConfiguring

    var isEncryptedTransportRequired: Bool {
        secureTransport.requiresEncryptedTransport
    }

    init(
        streamProtocol: StreamProtocol = StreamProtocol(),
        timeoutInterval: TimeInterval = 5,
        nowProvider: @escaping () -> Date = Date.init,
        secureTransport: any SecureTransportConfiguring = NetworkSecureTransportConfiguration()
    ) {
        self.streamProtocol = streamProtocol
        self.timeoutInterval = timeoutInterval
        self.nowProvider = nowProvider
        self.secureTransport = secureTransport
    }

    func startStreaming() {
        isStreaming = true
        _ = secureTransport.makeParameters()
    }

    func stopStreaming() {
        isStreaming = false
        isEncryptedChannelEstablished = false
    }

    func noteConnectionEstablished(at date: Date = Date()) {
        noteConnectionEstablished(at: date, encrypted: true)
    }

    func noteConnectionEstablished(at date: Date = Date(), encrypted: Bool) {
        do {
            try secureTransport.validateEncryptedChannelEstablished(encrypted)
            isEncryptedChannelEstablished = encrypted
            latestHeartbeatAt = date
            connectionState = .connected
        } catch {
            isEncryptedChannelEstablished = false
            connectionState = .failed(PairingFailureReason.encryptedChannelUnavailable.rawValue)
        }
    }

    func noteEncryptedConnectionEstablished(at date: Date = Date()) {
        latestHeartbeatAt = date
        connectionState = .connected
        isEncryptedChannelEstablished = true
    }

    func makeHeartbeatMessage(timestamp: UInt64) throws -> Data {
        try streamProtocol.encode(.heartbeat(timestamp: timestamp))
    }

    @discardableResult
    func receive(_ data: Data) throws -> StreamProtocol.Message {
        try secureTransport.validateEncryptedChannelEstablished(isEncryptedChannelEstablished)
        let message = try streamProtocol.decode(data)

        switch message {
        case .disconnectNotice:
            connectionState = .disconnected
        case .heartbeat, .videoFrame, .detectionResults:
            latestHeartbeatAt = nowProvider()
            connectionState = .connected
        }

        return message
    }

    func evaluateConnectionHealth(now: Date = Date()) {
        guard let latestHeartbeatAt else { return }

        let elapsed = now.timeIntervalSince(latestHeartbeatAt)
        if elapsed >= timeoutInterval {
            connectionState = .disconnected
        } else if elapsed >= timeoutInterval - 1 {
            connectionState = .stale
        } else {
            connectionState = .connected
        }
    }

    func registerFeed(_ session: PairingSession) {
        guard !feedStates.contains(where: { $0.sessionID == session.sessionID }) else { return }
        feedStates.append(
            SupervisorFeedTileState(
                sessionID: session.sessionID,
                deviceName: session.deviceName,
                connectionState: session.connectionState,
                latestFrame: nil,
                latestDetections: []
            )
        )
    }

    func publishFrame(_ frame: VideoFrame, for sessionID: UUID) {
        guard canSendFrameData() else { return }
        updateFeed(for: sessionID) { tile in
            tile.latestFrame = frame
            tile.connectionState = .connected
        }
    }

    func publishDetections(_ detections: [DetectionResult], for sessionID: UUID) {
        guard canSendFrameData() else { return }
        updateFeed(for: sessionID) { tile in
            tile.latestDetections = detections
            tile.connectionState = .connected
        }
    }

    func updateFeedConnectionState(_ state: ConnectionState, for sessionID: UUID) {
        updateFeed(for: sessionID) { tile in
            tile.connectionState = state
        }
    }

    private func updateFeed(for sessionID: UUID, update: (inout SupervisorFeedTileState) -> Void) {
        guard let index = feedStates.firstIndex(where: { $0.sessionID == sessionID }) else { return }
        var tile = feedStates[index]
        update(&tile)
        feedStates[index] = tile
    }

    private func canSendFrameData() -> Bool {
        do {
            try secureTransport.validateEncryptedChannelEstablished(isEncryptedChannelEstablished)
            return true
        } catch {
            connectionState = .failed(PairingFailureReason.encryptedChannelUnavailable.rawValue)
            return false
        }
    }
}
