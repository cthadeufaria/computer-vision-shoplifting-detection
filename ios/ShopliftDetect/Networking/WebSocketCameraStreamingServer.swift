import CryptoKit
import Foundation
@preconcurrency import Network

@MainActor
protocol CameraFrameBroadcasting: AnyObject {
    var isRunning: Bool { get }
    var connectedClientCount: Int { get }

    func start(session: PairingSession, token: PairingToken) throws
    func stop()
    func publish(frame: VideoFrame)
}

@MainActor
final class WebSocketCameraStreamingServer: CameraFrameBroadcasting {
    private struct Client {
        let id: UUID
        let connection: NWConnection
        var handshakeBuffer = Data()
        var isConnected = false
    }

    private struct VideoFrameMessage: Encodable {
        let type = "videoFrame"
        let timestamp: UInt64
        let width: Int
        let height: Int
        let jpegData: String
    }

    private var listener: NWListener?
    private var clients: [UUID: Client] = [:]
    private var expectedToken: String?

    private let encoder = JSONEncoder()

    var isRunning: Bool {
        listener != nil
    }

    var connectedClientCount: Int {
        clients.values.filter(\.isConnected).count
    }

    func start(session: PairingSession, token: PairingToken) throws {
        stop()

        guard let port = NWEndpoint.Port(rawValue: session.port) else {
            throw WebSocketCameraStreamingServerError.invalidPort
        }

        expectedToken = token.value

        let parameters = NWParameters.tcp
        parameters.allowLocalEndpointReuse = true

        let listener = try NWListener(using: parameters, on: port)
        listener.newConnectionHandler = { [weak self] connection in
            Task { @MainActor in
                self?.accept(connection)
            }
        }
        listener.stateUpdateHandler = { [weak self] state in
            if case .failed = state {
                Task { @MainActor in
                    self?.stop()
                }
            }
        }
        listener.start(queue: .main)
        self.listener = listener
    }

    func stop() {
        listener?.cancel()
        listener = nil
        expectedToken = nil

        for client in clients.values {
            client.connection.cancel()
        }
        clients.removeAll()
    }

    func publish(frame: VideoFrame) {
        guard !clients.isEmpty else { return }

        let message = VideoFrameMessage(
            timestamp: frame.timestamp,
            width: frame.width,
            height: frame.height,
            jpegData: frame.jpegData.base64EncodedString()
        )

        guard let payload = try? encoder.encode(message) else { return }
        let webSocketFrame = WebSocketFrameEncoder.encodeTextFrame(payload)

        for client in clients.values where client.isConnected {
            client.connection.send(content: webSocketFrame, completion: .contentProcessed { [weak self] error in
                guard error != nil else { return }
                Task { @MainActor in
                    self?.removeClient(client.id)
                }
            })
        }
    }

    private func accept(_ connection: NWConnection) {
        let clientID = UUID()
        clients[clientID] = Client(id: clientID, connection: connection)

        connection.stateUpdateHandler = { [weak self] state in
            switch state {
            case .failed, .cancelled:
                Task { @MainActor in
                    self?.removeClient(clientID)
                }
            default:
                break
            }
        }

        connection.start(queue: .main)
        receiveHandshake(for: clientID)
    }

    private func receiveHandshake(for clientID: UUID) {
        guard let client = clients[clientID] else { return }

        client.connection.receive(minimumIncompleteLength: 1, maximumLength: 4096) { [weak self] data, _, _, error in
            Task { @MainActor in
                guard let self else { return }
                guard error == nil, let data, !data.isEmpty else {
                    self.removeClient(clientID)
                    return
                }

                self.appendHandshakeData(data, for: clientID)
            }
        }
    }

    private func appendHandshakeData(_ data: Data, for clientID: UUID) {
        guard var client = clients[clientID] else { return }
        client.handshakeBuffer.append(data)
        clients[clientID] = client

        guard client.handshakeBuffer.range(of: Data("\r\n\r\n".utf8)) != nil else {
            receiveHandshake(for: clientID)
            return
        }

        do {
            let response = try makeHandshakeResponse(from: client.handshakeBuffer)
            client.connection.send(content: response, completion: .contentProcessed { [weak self] error in
                Task { @MainActor in
                    guard error == nil else {
                        self?.removeClient(clientID)
                        return
                    }

                    self?.markClientConnected(clientID)
                }
            })
        } catch {
            client.connection.cancel()
            removeClient(clientID)
        }
    }

    private func makeHandshakeResponse(from requestData: Data) throws -> Data {
        guard
            let request = String(data: requestData, encoding: .utf8),
            let requestLine = request.components(separatedBy: "\r\n").first,
            requestLine.hasPrefix("GET "),
            let path = requestLine.split(separator: " ").dropFirst().first
        else {
            throw WebSocketCameraStreamingServerError.invalidHandshake
        }

        guard requestContainsValidToken(path: String(path)) else {
            throw WebSocketCameraStreamingServerError.invalidToken
        }

        var headers: [String: String] = [:]
        for line in request.components(separatedBy: "\r\n").dropFirst() {
            guard let separator = line.firstIndex(of: ":") else { continue }
            let key = line[..<separator].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            let value = line[line.index(after: separator)...].trimmingCharacters(in: .whitespacesAndNewlines)
            headers[key] = value
        }

        guard
            headers["upgrade"]?.lowercased() == "websocket",
            headers["connection"]?.lowercased().contains("upgrade") == true,
            let webSocketKey = headers["sec-websocket-key"]
        else {
            throw WebSocketCameraStreamingServerError.invalidHandshake
        }

        let acceptValue = Self.acceptValue(for: webSocketKey)
        let response = """
        HTTP/1.1 101 Switching Protocols\r
        Upgrade: websocket\r
        Connection: Upgrade\r
        Sec-WebSocket-Accept: \(acceptValue)\r
        \r
        """

        return Data(response.utf8)
    }

    private func requestContainsValidToken(path: String) -> Bool {
        guard let expectedToken else { return false }
        guard let components = URLComponents(string: "ws://camera.local\(path)") else { return false }
        return components.queryItems?.first(where: { $0.name == "token" })?.value == expectedToken
    }

    private func markClientConnected(_ clientID: UUID) {
        guard var client = clients[clientID] else { return }
        client.isConnected = true
        clients[clientID] = client
    }

    private func removeClient(_ clientID: UUID) {
        clients[clientID]?.connection.cancel()
        clients.removeValue(forKey: clientID)
    }

    nonisolated static func acceptValue(for key: String) -> String {
        let magicGUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
        let digest = Insecure.SHA1.hash(data: Data((key + magicGUID).utf8))
        return Data(digest).base64EncodedString()
    }
}

enum WebSocketCameraStreamingServerError: LocalizedError, Equatable {
    case invalidPort
    case invalidHandshake
    case invalidToken

    var errorDescription: String? {
        switch self {
        case .invalidPort:
            return "The camera streaming port is invalid."
        case .invalidHandshake:
            return "The browser did not send a valid WebSocket handshake."
        case .invalidToken:
            return "The browser pairing token is invalid."
        }
    }
}

enum WebSocketFrameEncoder {
    static func encodeTextFrame(_ payload: Data) -> Data {
        var frame = Data([0x81])

        if payload.count <= 125 {
            frame.append(UInt8(payload.count))
        } else if payload.count <= UInt16.max {
            frame.append(126)
            appendUInt16(UInt16(payload.count), to: &frame)
        } else {
            frame.append(127)
            appendUInt64(UInt64(payload.count), to: &frame)
        }

        frame.append(payload)
        return frame
    }

    private static func appendUInt16(_ value: UInt16, to data: inout Data) {
        var bigEndian = value.bigEndian
        data.append(Data(bytes: &bigEndian, count: MemoryLayout<UInt16>.size))
    }

    private static func appendUInt64(_ value: UInt64, to data: inout Data) {
        var bigEndian = value.bigEndian
        data.append(Data(bytes: &bigEndian, count: MemoryLayout<UInt64>.size))
    }
}
