import Foundation
import XCTest
@testable import ShopliftDetect

final class WebSocketCameraStreamingServerTests: XCTestCase {
    func test_acceptValue_matchesRFCExample() {
        XCTAssertEqual(
            WebSocketCameraStreamingServer.acceptValue(for: "dGhlIHNhbXBsZSBub25jZQ=="),
            "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
        )
    }

    func test_encodeTextFrame_usesSmallPayloadLength() {
        let frame = WebSocketFrameEncoder.encodeTextFrame(Data("hello".utf8))

        XCTAssertEqual(frame[0], 0x81)
        XCTAssertEqual(frame[1], 5)
        XCTAssertEqual(String(data: frame.dropFirst(2), encoding: .utf8), "hello")
    }

    func test_encodeTextFrame_usesExtendedPayloadLength() {
        let payload = Data(repeating: 0x41, count: 126)
        let frame = WebSocketFrameEncoder.encodeTextFrame(payload)

        XCTAssertEqual(frame[0], 0x81)
        XCTAssertEqual(frame[1], 126)
        XCTAssertEqual(frame[2], 0)
        XCTAssertEqual(frame[3], 126)
        XCTAssertEqual(frame.dropFirst(4).count, 126)
    }
}
