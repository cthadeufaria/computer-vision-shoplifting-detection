import XCTest
@testable import ShopliftDetect

@MainActor
final class SupervisorViewModelTests: XCTestCase {
    func test_refreshLoadsTilesFromStreamingService() {
        let pairing = MockPairingService()
        let streaming = MockStreamingService()
        let session = PairingSession(
            sessionID: UUID(uuidString: "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE")!,
            role: .camera,
            deviceName: "Aisle 3 Camera",
            host: "192.168.1.24",
            port: 7890,
            connectionState: .connected
        )
        let tile = SupervisorFeedTileState(
            sessionID: session.sessionID,
            deviceName: session.deviceName,
            connectionState: .connected,
            latestFrame: VideoFrame(timestamp: 10, jpegData: Data([0x01]), width: 64, height: 64),
            latestDetections: []
        )
        pairing.sessions = [session]
        streaming.feedStates = [tile]

        let sut = SupervisorViewModel(pairing: pairing, streaming: streaming)
        sut.refresh()

        XCTAssertEqual(sut.grid.tiles.count, 1)
        XCTAssertEqual(sut.grid.tiles.first?.deviceName, "Aisle 3 Camera")
    }

    func test_selectTile_setsFullScreenSelection() {
        let sut = SupervisorViewModel(pairing: MockPairingService(), streaming: MockStreamingService())
        let tile = SupervisorFeedTileState(
            sessionID: UUID(),
            deviceName: "Aisle 3 Camera",
            connectionState: .connected,
            latestFrame: nil,
            latestDetections: []
        )

        sut.select(tile)

        XCTAssertEqual(sut.selectedTile?.sessionID, tile.sessionID)
    }

    func test_refreshPreservesStaleTileState() {
        let pairing = MockPairingService()
        let streaming = MockStreamingService()
        let session = PairingSession(
            sessionID: UUID(uuidString: "11111111-2222-3333-4444-555555555555")!,
            role: .camera,
            deviceName: "Aisle 5 Camera",
            host: "192.168.1.26",
            port: 7891,
            connectionState: .stale
        )
        pairing.sessions = [session]
        streaming.feedStates = [
            SupervisorFeedTileState(
                sessionID: session.sessionID,
                deviceName: session.deviceName,
                connectionState: .stale,
                latestFrame: VideoFrame(timestamp: 11, jpegData: Data([0xFF]), width: 64, height: 64),
                latestDetections: []
            )
        ]

        let sut = SupervisorViewModel(pairing: pairing, streaming: streaming)
        sut.refresh()

        XCTAssertEqual(sut.grid.tiles.first?.connectionState, .stale)
        XCTAssertNotNil(sut.grid.tiles.first?.latestFrame)
    }
}
