import Darwin
import Foundation

@MainActor
enum PairingFailureReason: String, Error, Equatable, Sendable {
    case invalidPayload = "invalid_payload"
    case invalidToken = "invalid_token"
    case expiredToken = "expired_token"
    case reusedToken = "reused_token"
    case unsupportedVersion = "unsupported_version"
    case connectionLimitReached = "connection_limit_reached"
    case encryptedChannelUnavailable = "encrypted_channel_unavailable"
}

struct PairingPayload: Equatable, Sendable {
    let host: String
    let port: UInt16
    let token: String

    var rawValue: String {
        "sdlink://\(host):\(port)?token=\(token)"
    }
}

@MainActor
final class PairingService: PairingServiceProtocol {
    private(set) var connectionState: ConnectionState = .idle
    private(set) var currentSession: PairingSession?
    private(set) var currentToken: PairingToken?
    private(set) var sessions: [PairingSession] = []

    var qrPayloadString: String? {
        guard
            let session = currentSession,
            let token = currentToken,
            token.isVisibleOnScreen
        else {
            return nil
        }

        return PairingPayload(
            host: session.host,
            port: session.port,
            token: token.value
        ).rawValue
    }

    private let lanHostProvider: () -> String
    private let portProvider: () -> UInt16
    private let tokenProvider: () -> String
    private let nowProvider: () -> Date
    private let externalValidationToken: String?
    private let secureTransport: any SecureTransportConfiguring
    private let encryptedChannelEstablishedProvider: () -> Bool

    var isEncryptedTransportRequired: Bool {
        secureTransport.requiresEncryptedTransport
    }

    init(
        lanHostProvider: @escaping () -> String = PairingService.currentLANHost,
        portProvider: @escaping () -> UInt16 = { 7890 },
        tokenProvider: @escaping () -> String = { UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(8).uppercased() },
        nowProvider: @escaping () -> Date = Date.init,
        externalValidationToken: String? = nil,
        secureTransport: any SecureTransportConfiguring = NetworkSecureTransportConfiguration(),
        encryptedChannelEstablishedProvider: @escaping () -> Bool = { true }
    ) {
        self.lanHostProvider = lanHostProvider
        self.portProvider = portProvider
        self.tokenProvider = tokenProvider
        self.nowProvider = nowProvider
        self.externalValidationToken = externalValidationToken
        self.secureTransport = secureTransport
        self.encryptedChannelEstablishedProvider = encryptedChannelEstablishedProvider
    }

    @discardableResult
    func prepareCameraPairing(deviceName: String) -> String? {
        let token = PairingToken(
            value: tokenProvider(),
            issuedAt: nowProvider(),
            isConsumed: false,
            isVisibleOnScreen: true
        )
        currentToken = token
        currentSession = PairingSession(
            role: .camera,
            deviceName: deviceName,
            host: lanHostProvider(),
            port: portProvider(),
            connectionState: .listening,
            token: token
        )
        connectionState = .listening
        return qrPayloadString
    }

    func prepareSupervisorPairing() {
        currentSession = nil
        currentToken = nil
        connectionState = .idle
    }

    func connectToCamera(using payloadString: String, deviceName: String) {
        do {
            connectionState = .connecting
            let payload = try Self.parsePayload(payloadString)
            connectionState = .handshaking
            _ = secureTransport.makeParameters()
            try secureTransport.validateEncryptedChannelEstablished(encryptedChannelEstablishedProvider())
            try validateToken(for: payload)
            try validateSupervisorCapacity()

            let session = PairingSession(
                role: .supervisor,
                deviceName: deviceName,
                host: payload.host,
                port: payload.port,
                connectionState: .connected,
                heartbeatDeadline: nowProvider().addingTimeInterval(5),
                token: PairingToken(
                    value: payload.token,
                    issuedAt: nowProvider(),
                    isConsumed: true,
                    isVisibleOnScreen: false
                )
            )
            currentSession = session
            sessions.append(session)
            connectionState = .connected
        } catch let reason as PairingFailureReason {
            currentSession?.connectionState = .failed(reason.rawValue)
            connectionState = .failed(reason.rawValue)
        } catch SecureTransportError.encryptedChannelUnavailable {
            currentSession?.connectionState = .failed(PairingFailureReason.encryptedChannelUnavailable.rawValue)
            connectionState = .failed(PairingFailureReason.encryptedChannelUnavailable.rawValue)
        } catch {
            currentSession?.connectionState = .failed(PairingFailureReason.invalidPayload.rawValue)
            connectionState = .failed(PairingFailureReason.invalidPayload.rawValue)
        }
    }

    func expireCameraPairing() {
        if var token = currentToken {
            token.isVisibleOnScreen = false
            currentToken = token
        }

        if currentSession?.role == .camera {
            currentSession?.connectionState = .idle
        } else {
            currentSession = nil
        }
        connectionState = .idle
    }

    func reset() {
        currentSession = nil
        currentToken = nil
        sessions = []
        connectionState = .idle
    }

    func seedSupervisorSessions(deviceNames: [String]) {
        sessions = deviceNames.enumerated().map { index, deviceName in
            PairingSession(
                role: .supervisor,
                deviceName: deviceName,
                host: "192.168.1.\(20 + index)",
                port: UInt16(7890 + index),
                connectionState: .connected
            )
        }
    }

    static func parsePayload(_ rawValue: String) throws -> PairingPayload {
        guard
            let components = URLComponents(string: rawValue),
            components.scheme == "sdlink",
            let host = components.host,
            let port = components.port,
            let token = components.queryItems?.first(where: { $0.name == "token" })?.value,
            !token.isEmpty,
            Self.isLANReachableHost(host)
        else {
            throw PairingFailureReason.invalidPayload
        }

        return PairingPayload(host: host, port: UInt16(port), token: token)
    }

    private func validateToken(for payload: PairingPayload) throws {
        if let currentToken {
            guard currentToken.isVisibleOnScreen else {
                throw PairingFailureReason.expiredToken
            }
            guard !currentToken.isConsumed else {
                throw PairingFailureReason.reusedToken
            }
            guard currentToken.value == payload.token else {
                throw PairingFailureReason.invalidToken
            }

            var consumedToken = currentToken
            consumedToken.isConsumed = true
            consumedToken.isVisibleOnScreen = false
            self.currentToken = consumedToken
        } else if let externalValidationToken, payload.token != externalValidationToken {
            throw PairingFailureReason.invalidToken
        }
    }

    private func validateSupervisorCapacity() throws {
        guard sessions.count < SupervisorFeedGrid.maxFeeds else {
            throw PairingFailureReason.connectionLimitReached
        }
    }

    nonisolated private static func isLANReachableHost(_ host: String) -> Bool {
        if host.hasSuffix(".local") {
            return true
        }

        let octets = host.split(separator: ".").compactMap { Int($0) }
        guard octets.count == 4 else {
            return false
        }

        let first = octets[0]
        let second = octets[1]

        if first == 10 {
            return true
        }
        if first == 172, (16...31).contains(second) {
            return true
        }
        if first == 192, second == 168 {
            return true
        }

        return false
    }

    nonisolated private static func currentLANHost() -> String {
        var interfaces: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&interfaces) == 0, let firstInterface = interfaces else {
            return "127.0.0.1"
        }
        defer { freeifaddrs(interfaces) }

        var fallbackHost: String?
        var cursor: UnsafeMutablePointer<ifaddrs>? = firstInterface

        while let interface = cursor?.pointee {
            defer { cursor = interface.ifa_next }

            guard
                let address = interface.ifa_addr,
                Int32(address.pointee.sa_family) == AF_INET,
                isUsableLocalInterface(interface),
                let host = numericHost(for: address),
                isLANReachableHost(host)
            else {
                continue
            }

            let interfaceName = String(cString: interface.ifa_name)
            if interfaceName == "en0" {
                return host
            }

            fallbackHost = fallbackHost ?? host
        }

        return fallbackHost ?? "127.0.0.1"
    }

    nonisolated private static func isUsableLocalInterface(_ interface: ifaddrs) -> Bool {
        let flags = interface.ifa_flags
        let isUp = (flags & UInt32(IFF_UP)) != 0
        let isLoopback = (flags & UInt32(IFF_LOOPBACK)) != 0
        return isUp && !isLoopback
    }

    nonisolated private static func numericHost(for address: UnsafePointer<sockaddr>) -> String? {
        var hostname = [CChar](repeating: 0, count: Int(NI_MAXHOST))
        let result = getnameinfo(
            address,
            socklen_t(MemoryLayout<sockaddr_in>.size),
            &hostname,
            socklen_t(hostname.count),
            nil,
            0,
            NI_NUMERICHOST
        )

        guard result == 0 else { return nil }
        let endIndex = hostname.firstIndex(of: 0) ?? hostname.endIndex
        let bytes = hostname[..<endIndex].map { UInt8(bitPattern: $0) }
        return String(decoding: bytes, as: UTF8.self)
    }
}
