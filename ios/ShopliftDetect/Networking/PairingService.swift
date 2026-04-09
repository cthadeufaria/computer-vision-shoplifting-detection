import Foundation

@MainActor
final class NoopPairingService: PairingServiceProtocol {
    private(set) var connectionState: ConnectionState = .idle

    func reset() {
        connectionState = .idle
    }
}
