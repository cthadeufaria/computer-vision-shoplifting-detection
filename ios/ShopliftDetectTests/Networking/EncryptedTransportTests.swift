import Network
import XCTest
@testable import ShopliftDetect

@MainActor
final class EncryptedTransportTests: XCTestCase {
    func test_secureTransportParameters_includeTLSOptions() {
        let sut = NetworkSecureTransportConfiguration()

        let parameters = sut.makeParameters()

        XCTAssertTrue(sut.requiresEncryptedTransport)
        XCTAssertTrue(
            parameters.defaultProtocolStack.applicationProtocols.contains { $0 is NWProtocolTLS.Options }
        )
    }

    func test_connectToCamera_whenEncryptedChannelFails_rejectsBeforeConsumingToken() {
        let sut = PairingService(
            lanHostProvider: { "192.168.1.24" },
            portProvider: { 7890 },
            tokenProvider: { "ABCD1234" },
            encryptedChannelEstablishedProvider: { false }
        )
        let payload = sut.prepareCameraPairing(deviceName: "Aisle Camera") ?? ""

        sut.connectToCamera(using: payload, deviceName: "Front Desk iPad")

        XCTAssertEqual(sut.connectionState, .failed(PairingFailureReason.encryptedChannelUnavailable.rawValue))
        XCTAssertEqual(sut.currentToken?.isConsumed, false)
        XCTAssertEqual(sut.sessions, [])
    }

    func test_streamingRejectsFrameBeforeEncryptedConnectionIsEstablished() {
        let session = PairingSession(
            role: .supervisor,
            deviceName: "Aisle Camera",
            host: "192.168.1.24",
            port: 7890,
            connectionState: .connected
        )
        let frame = VideoFrame(timestamp: 1, jpegData: Data([0x01]), width: 120, height: 90)
        let sut = StreamingService()

        sut.registerFeed(session)
        sut.publishFrame(frame, for: session.sessionID)

        XCTAssertNil(sut.feedStates.first?.latestFrame)
        XCTAssertEqual(sut.connectionState, .failed(PairingFailureReason.encryptedChannelUnavailable.rawValue))
    }

    func test_streamingPublishesFrameAfterEncryptedConnectionIsEstablished() {
        let session = PairingSession(
            role: .supervisor,
            deviceName: "Aisle Camera",
            host: "192.168.1.24",
            port: 7890,
            connectionState: .connected
        )
        let frame = VideoFrame(timestamp: 1, jpegData: Data([0x01]), width: 120, height: 90)
        let sut = StreamingService()

        sut.registerFeed(session)
        sut.noteConnectionEstablished(at: Date(timeIntervalSince1970: 100), encrypted: true)
        sut.publishFrame(frame, for: session.sessionID)

        XCTAssertEqual(sut.feedStates.first?.latestFrame, frame)
        XCTAssertEqual(sut.connectionState, .connected)
    }

    func test_receiveBeforeEncryptedConnectionIsEstablished_throws() throws {
        let sut = StreamingService()
        let heartbeat = try StreamProtocol().encode(.heartbeat(timestamp: 100))

        XCTAssertThrowsError(try sut.receive(heartbeat)) { error in
            XCTAssertEqual(error as? SecureTransportError, .encryptedChannelUnavailable)
        }
    }
}
