import Foundation

@MainActor
protocol StreamingServiceProtocol: AnyObject {
    var isStreaming: Bool { get }
    var connectionState: ConnectionState { get }
    var latestHeartbeatAt: Date? { get }
    var feedStates: [SupervisorFeedTileState] { get }

    func startStreaming()
    func stopStreaming()
    func noteConnectionEstablished(at date: Date)
    func makeHeartbeatMessage(timestamp: UInt64) throws -> Data
    @discardableResult
    func receive(_ data: Data) throws -> StreamProtocol.Message
    func evaluateConnectionHealth(now: Date)
    func registerFeed(_ session: PairingSession)
    func publishFrame(_ frame: VideoFrame, for sessionID: UUID)
    func publishDetections(_ detections: [DetectionResult], for sessionID: UUID)
    func updateFeedConnectionState(_ state: ConnectionState, for sessionID: UUID)
}
