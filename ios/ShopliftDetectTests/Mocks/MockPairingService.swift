@testable import ShopliftDetect

@MainActor
final class MockPairingService: PairingServiceProtocol {
    var connectionState: ConnectionState = .idle
    var currentSession: PairingSession?
    var currentToken: PairingToken?
    var qrPayloadString: String?
    var sessions: [PairingSession] = []
    private(set) var resetCallCount = 0
    private(set) var prepareCameraPairingCallCount = 0
    private(set) var prepareSupervisorPairingCallCount = 0
    private(set) var connectCallCount = 0
    private(set) var expireCameraPairingCallCount = 0
    private(set) var lastConnectedPayload: String?

    @discardableResult
    func prepareCameraPairing(deviceName: String) -> String? {
        prepareCameraPairingCallCount += 1
        if qrPayloadString == nil {
            qrPayloadString = "sdlink://192.168.1.24:7890?token=TEST1234"
        }
        currentToken = PairingToken(value: "TEST1234")
        currentSession = PairingSession(
            role: .camera,
            deviceName: deviceName,
            host: "192.168.1.24",
            port: 7890,
            connectionState: .listening,
            token: currentToken
        )
        if let currentSession {
            sessions = [currentSession]
        }
        connectionState = .listening
        return qrPayloadString
    }

    func prepareSupervisorPairing() {
        prepareSupervisorPairingCallCount += 1
        connectionState = .idle
    }

    func connectToCamera(using payloadString: String, deviceName: String) {
        connectCallCount += 1
        lastConnectedPayload = payloadString
        currentSession = PairingSession(
            role: .supervisor,
            deviceName: deviceName,
            host: "192.168.1.24",
            port: 7890,
            connectionState: .connected
        )
        if let currentSession {
            sessions.append(currentSession)
        }
        connectionState = .connected
    }

    func expireCameraPairing() {
        expireCameraPairingCallCount += 1
        if var currentToken {
            currentToken.isVisibleOnScreen = false
            self.currentToken = currentToken
        }
        connectionState = .idle
    }

    func reset() {
        resetCallCount += 1
        connectionState = .idle
        currentSession = nil
        currentToken = nil
        qrPayloadString = nil
        sessions = []
    }
}
