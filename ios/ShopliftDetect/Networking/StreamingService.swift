import Foundation

@MainActor
final class StreamingService: StreamingServiceProtocol {
    private(set) var isStreaming = false
    private(set) var connectionState: ConnectionState = .idle
    private(set) var latestHeartbeatAt: Date?
    private(set) var feedStates: [SupervisorFeedTileState] = []

    private let streamProtocol: StreamProtocol
    private let nowProvider: () -> Date
    private let timeoutInterval: TimeInterval

    init(
        streamProtocol: StreamProtocol = StreamProtocol(),
        timeoutInterval: TimeInterval = 5,
        nowProvider: @escaping () -> Date = Date.init
    ) {
        self.streamProtocol = streamProtocol
        self.timeoutInterval = timeoutInterval
        self.nowProvider = nowProvider
    }

    func startStreaming() {
        isStreaming = true
    }

    func stopStreaming() {
        isStreaming = false
    }

    func noteConnectionEstablished(at date: Date = Date()) {
        latestHeartbeatAt = date
        connectionState = .connected
    }

    func makeHeartbeatMessage(timestamp: UInt64) throws -> Data {
        try streamProtocol.encode(.heartbeat(timestamp: timestamp))
    }

    @discardableResult
    func receive(_ data: Data) throws -> StreamProtocol.Message {
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
        updateFeed(for: sessionID) { tile in
            tile.latestFrame = frame
            tile.connectionState = .connected
        }
    }

    func publishDetections(_ detections: [DetectionResult], for sessionID: UUID) {
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
}
