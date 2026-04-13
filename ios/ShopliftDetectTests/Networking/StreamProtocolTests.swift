import XCTest
@testable import ShopliftDetect

@MainActor
final class StreamProtocolTests: XCTestCase {
    func test_encodeAndDecodeHeartbeatEnvelope_roundTrips() throws {
        let sut = StreamProtocol()

        let data = try sut.encode(.heartbeat(timestamp: 1_775_700_000))
        let decoded = try sut.decode(data)

        XCTAssertEqual(decoded, .heartbeat(timestamp: 1_775_700_000))
    }

    func test_encodeAndDecodeVideoFrame_roundTrips() throws {
        let sut = StreamProtocol()
        let frame = VideoFrame(
            timestamp: 99,
            jpegData: Data([0x01, 0x02, 0x03]),
            width: 320,
            height: 180
        )

        let data = try sut.encode(.videoFrame(frame))
        let decoded = try sut.decode(data)

        XCTAssertEqual(decoded, .videoFrame(frame))
    }

    func test_receiveHeartbeat_updatesStreamingStateToConnected() throws {
        let streaming = StreamingService(nowProvider: { Date(timeIntervalSince1970: 100) })
        let heartbeat = try StreamProtocol().encode(.heartbeat(timestamp: 100))

        streaming.noteConnectionEstablished(at: Date(timeIntervalSince1970: 99), encrypted: true)
        let message = try streaming.receive(heartbeat)

        XCTAssertEqual(message, .heartbeat(timestamp: 100))
        XCTAssertEqual(streaming.connectionState, .connected)
        XCTAssertEqual(streaming.latestHeartbeatAt, Date(timeIntervalSince1970: 100))
    }

    func test_evaluateConnectionHealth_marksStaleThenDisconnected() {
        let streaming = StreamingService()
        let connectedAt = Date(timeIntervalSince1970: 100)
        streaming.noteConnectionEstablished(at: connectedAt)

        streaming.evaluateConnectionHealth(now: connectedAt.addingTimeInterval(4.2))
        XCTAssertEqual(streaming.connectionState, .stale)

        streaming.evaluateConnectionHealth(now: connectedAt.addingTimeInterval(5.1))
        XCTAssertEqual(streaming.connectionState, .disconnected)
    }
}
