import Foundation

@MainActor
protocol PairingServiceProtocol: AnyObject {
    var connectionState: ConnectionState { get }
    func reset()
}
