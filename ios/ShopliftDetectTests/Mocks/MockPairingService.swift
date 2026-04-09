@testable import ShopliftDetect

@MainActor
final class MockPairingService: PairingServiceProtocol {
    var connectionState: ConnectionState = .idle
    private(set) var resetCallCount = 0

    func reset() {
        resetCallCount += 1
        connectionState = .idle
    }
}
