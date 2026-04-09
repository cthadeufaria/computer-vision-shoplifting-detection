import Foundation

@MainActor
final class SupervisorViewModel: ObservableObject {
    @Published private(set) var grid = SupervisorFeedGrid()
    @Published private(set) var selectedTile: SupervisorFeedTileState?

    private let pairing: PairingServiceProtocol
    private let streaming: StreamingServiceProtocol

    init(
        pairing: PairingServiceProtocol,
        streaming: StreamingServiceProtocol
    ) {
        self.pairing = pairing
        self.streaming = streaming
    }

    var showsLimitBanner: Bool {
        pairing.sessions.count >= SupervisorFeedGrid.maxFeeds
    }

    func refresh() {
        for session in pairing.sessions {
            streaming.registerFeed(session)
        }

        let sessionNames = Dictionary(uniqueKeysWithValues: pairing.sessions.map { ($0.sessionID, $0.deviceName) })
        let tiles = streaming.feedStates.map { tile in
            var tile = tile
            if let sessionName = sessionNames[tile.sessionID] {
                tile = SupervisorFeedTileState(
                    sessionID: tile.sessionID,
                    deviceName: sessionName,
                    connectionState: tile.connectionState,
                    latestFrame: tile.latestFrame,
                    latestDetections: tile.latestDetections
                )
            }
            return tile
        }
        grid = SupervisorFeedGrid(tiles: tiles)

        if let selectedTile {
            self.selectedTile = tiles.first(where: { $0.sessionID == selectedTile.sessionID })
        }
    }

    func select(_ tile: SupervisorFeedTileState) {
        selectedTile = tile
    }

    func clearSelection() {
        selectedTile = nil
    }
}
