import Foundation

enum ConnectionState: Equatable, Sendable {
    case idle
    case listening
    case connecting
    case handshaking
    case connected
    case stale
    case disconnected
    case failed(String)
}
