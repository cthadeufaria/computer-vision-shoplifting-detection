import Foundation
import Network

enum SecureTransportError: Error, Equatable, Sendable {
    case encryptedChannelUnavailable
}

protocol SecureTransportConfiguring: Sendable {
    var requiresEncryptedTransport: Bool { get }
    func makeParameters() -> NWParameters
    func validateEncryptedChannelEstablished(_ isEstablished: Bool) throws
}

struct NetworkSecureTransportConfiguration: SecureTransportConfiguring {
    let requiresEncryptedTransport: Bool

    init(requiresEncryptedTransport: Bool = true) {
        self.requiresEncryptedTransport = requiresEncryptedTransport
    }

    func makeParameters() -> NWParameters {
        let tlsOptions = NWProtocolTLS.Options()
        let tcpOptions = NWProtocolTCP.Options()
        let parameters = NWParameters(tls: tlsOptions, tcp: tcpOptions)
        parameters.includePeerToPeer = true
        return parameters
    }

    func validateEncryptedChannelEstablished(_ isEstablished: Bool) throws {
        guard !requiresEncryptedTransport || isEstablished else {
            throw SecureTransportError.encryptedChannelUnavailable
        }
    }
}
