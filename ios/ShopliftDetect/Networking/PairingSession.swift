import Foundation

struct PairingSession: Equatable, Sendable {
    let sessionID: UUID
    let role: DeviceRole
    let deviceName: String
    let host: String
    let port: UInt16
    var connectionState: ConnectionState
    var heartbeatDeadline: Date?
    var token: PairingToken?

    init(
        sessionID: UUID = UUID(),
        role: DeviceRole,
        deviceName: String,
        host: String,
        port: UInt16,
        connectionState: ConnectionState = .idle,
        heartbeatDeadline: Date? = nil,
        token: PairingToken? = nil
    ) {
        self.sessionID = sessionID
        self.role = role
        self.deviceName = deviceName
        self.host = host
        self.port = port
        self.connectionState = connectionState
        self.heartbeatDeadline = heartbeatDeadline
        self.token = token
    }
}
