import Foundation
@testable import ShopliftDetect

@MainActor
final class MockStreamingService: StreamingServiceProtocol {
    var isStreaming = false
    var connectionState: ConnectionState = .idle
    var latestHeartbeatAt: Date?
    var feedStates: [SupervisorFeedTileState] = []
    private(set) var startCallCount = 0
    private(set) var stopCallCount = 0
    private(set) var noteConnectionEstablishedCallCount = 0
    private(set) var receiveCallCount = 0

    func startStreaming() {
        startCallCount += 1
        isStreaming = true
    }

    func stopStreaming() {
        stopCallCount += 1
        isStreaming = false
    }

    func noteConnectionEstablished(at date: Date) {
        noteConnectionEstablishedCallCount += 1
        latestHeartbeatAt = date
        connectionState = .connected
    }

    func makeHeartbeatMessage(timestamp: UInt64) throws -> Data {
        try StreamProtocol().encode(.heartbeat(timestamp: timestamp))
    }

    func receive(_ data: Data) throws -> StreamProtocol.Message {
        receiveCallCount += 1
        return try StreamProtocol().decode(data)
    }

    func evaluateConnectionHealth(now: Date) {
        if let latestHeartbeatAt, now.timeIntervalSince(latestHeartbeatAt) >= 5 {
            connectionState = .disconnected
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
        guard let index = feedStates.firstIndex(where: { $0.sessionID == sessionID }) else { return }
        feedStates[index].latestFrame = frame
        feedStates[index].connectionState = .connected
    }

    func publishDetections(_ detections: [DetectionResult], for sessionID: UUID) {
        guard let index = feedStates.firstIndex(where: { $0.sessionID == sessionID }) else { return }
        feedStates[index].latestDetections = detections
        feedStates[index].connectionState = .connected
    }

    func updateFeedConnectionState(_ state: ConnectionState, for sessionID: UUID) {
        guard let index = feedStates.firstIndex(where: { $0.sessionID == sessionID }) else { return }
        feedStates[index].connectionState = state
    }
}
