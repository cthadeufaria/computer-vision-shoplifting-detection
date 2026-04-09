import Foundation

@MainActor
protocol PairingServiceProtocol: AnyObject {
    var connectionState: ConnectionState { get }
    var currentSession: PairingSession? { get }
    var currentToken: PairingToken? { get }
    var qrPayloadString: String? { get }
    var sessions: [PairingSession] { get }

    @discardableResult
    func prepareCameraPairing(deviceName: String) -> String?
    func prepareSupervisorPairing()
    func connectToCamera(using payloadString: String, deviceName: String)
    func expireCameraPairing()
    func reset()
}
