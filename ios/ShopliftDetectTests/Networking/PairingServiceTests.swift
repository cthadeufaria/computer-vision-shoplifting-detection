import XCTest
@testable import ShopliftDetect

@MainActor
final class PairingServiceTests: XCTestCase {
    func test_prepareCameraPairing_generatesExpectedQRPayload() {
        let sut = PairingService(
            lanHostProvider: { "192.168.1.24" },
            portProvider: { 7890 },
            tokenProvider: { "ABCD1234" }
        )

        let payload = sut.prepareCameraPairing(deviceName: "Aisle Camera")

        XCTAssertEqual(payload, "sdlink://192.168.1.24:7890?token=ABCD1234")
        XCTAssertEqual(sut.connectionState, .listening)
        XCTAssertEqual(sut.currentToken?.value, "ABCD1234")
        XCTAssertEqual(sut.currentToken?.isVisibleOnScreen, true)
    }

    func test_connectToCamera_withVisibleMatchingToken_marksConnectedAndConsumesToken() {
        let sut = PairingService(
            lanHostProvider: { "192.168.1.24" },
            portProvider: { 7890 },
            tokenProvider: { "ABCD1234" }
        )
        let payload = sut.prepareCameraPairing(deviceName: "Aisle Camera") ?? ""

        sut.connectToCamera(using: payload, deviceName: "Front Desk iPad")

        XCTAssertEqual(sut.connectionState, .connected)
        XCTAssertEqual(sut.currentSession?.role, .supervisor)
        XCTAssertEqual(sut.currentToken?.isConsumed, true)
        XCTAssertEqual(sut.currentToken?.isVisibleOnScreen, false)
    }

    func test_connectToCamera_afterTokenExpired_failsWithExpiredToken() {
        let sut = PairingService(
            lanHostProvider: { "192.168.1.24" },
            portProvider: { 7890 },
            tokenProvider: { "ABCD1234" }
        )
        let payload = sut.prepareCameraPairing(deviceName: "Aisle Camera") ?? ""
        sut.expireCameraPairing()

        sut.connectToCamera(using: payload, deviceName: "Front Desk iPad")

        XCTAssertEqual(sut.connectionState, .failed(PairingFailureReason.expiredToken.rawValue))
    }

    func test_connectToCamera_withMalformedPayload_failsValidation() {
        let sut = PairingService()

        sut.connectToCamera(using: "https://example.com?token=bad", deviceName: "Front Desk iPad")

        XCTAssertEqual(sut.connectionState, .failed(PairingFailureReason.invalidPayload.rawValue))
    }

    func test_connectToCamera_withExternalValidationToken_rejectsInvalidToken() {
        let sut = PairingService(externalValidationToken: "VALID123")

        sut.connectToCamera(
            using: "sdlink://192.168.1.24:7890?token=WRONG999",
            deviceName: "Front Desk iPad"
        )

        XCTAssertEqual(sut.connectionState, .failed(PairingFailureReason.invalidToken.rawValue))
    }

    func test_parsePayload_rejectsNonLanHost() {
        XCTAssertThrowsError(try PairingService.parsePayload("sdlink://8.8.8.8:7890?token=ABCD1234"))
    }

    func test_connectToCamera_whenSupervisorAlreadyHasFourSessions_failsWithConnectionLimit() {
        let sut = PairingService(externalValidationToken: "VALID123")
        sut.seedSupervisorSessions(deviceNames: ["One", "Two", "Three", "Four"])

        sut.connectToCamera(
            using: "sdlink://192.168.1.24:7890?token=VALID123",
            deviceName: "Front Desk iPad"
        )

        XCTAssertEqual(sut.connectionState, .failed(PairingFailureReason.connectionLimitReached.rawValue))
        XCTAssertEqual(sut.sessions.count, 4)
    }

    func test_connectToCamera_afterExpiredToken_generatesFreshPayloadThatConnects() {
        final class TokenBox: @unchecked Sendable {
            var values = ["OLDTOKEN", "NEWTOKEN"]
        }

        let tokenBox = TokenBox()
        let sut = PairingService(
            lanHostProvider: { "192.168.1.24" },
            portProvider: { 7890 },
            tokenProvider: { tokenBox.values.removeFirst() }
        )

        let expiredPayload = sut.prepareCameraPairing(deviceName: "Aisle Camera") ?? ""
        sut.expireCameraPairing()
        sut.connectToCamera(using: expiredPayload, deviceName: "Front Desk iPad")
        XCTAssertEqual(sut.connectionState, .failed(PairingFailureReason.expiredToken.rawValue))

        let freshPayload = sut.prepareCameraPairing(deviceName: "Aisle Camera") ?? ""
        sut.connectToCamera(using: freshPayload, deviceName: "Front Desk iPad")

        XCTAssertEqual(sut.connectionState, .connected)
        XCTAssertEqual(sut.currentToken?.value, "NEWTOKEN")
    }

    func test_connectToCamera_afterInvalidToken_canRecoverWithValidPayload() {
        let sut = PairingService(externalValidationToken: "VALID123")

        sut.connectToCamera(
            using: "sdlink://192.168.1.24:7890?token=WRONG999",
            deviceName: "Front Desk iPad"
        )
        XCTAssertEqual(sut.connectionState, .failed(PairingFailureReason.invalidToken.rawValue))

        sut.connectToCamera(
            using: "sdlink://192.168.1.24:7890?token=VALID123",
            deviceName: "Front Desk iPad"
        )

        XCTAssertEqual(sut.connectionState, .connected)
        XCTAssertEqual(sut.currentSession?.connectionState, .connected)
    }
}
