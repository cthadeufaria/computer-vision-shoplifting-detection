import Foundation

@MainActor
final class SupervisorViewModel: ObservableObject {
    @Published private(set) var grid = SupervisorFeedGrid()
    @Published private(set) var selectedTile: SupervisorFeedTileState?

    private let pairing: PairingServiceProtocol
    private let streaming: StreamingServiceProtocol
    private let remoteInference: RemoteInferenceService

    init(
        pairing: PairingServiceProtocol,
        streaming: StreamingServiceProtocol,
        remoteInference: RemoteInferenceService
    ) {
        self.pairing = pairing
        self.streaming = streaming
        self.remoteInference = remoteInference
    }

    var showsLimitBanner: Bool {
        pairing.sessions.count >= SupervisorFeedGrid.maxFeeds
    }

    func refresh() async {
        for session in pairing.sessions {
            streaming.registerFeed(session)
        }

        let sessionNames = Dictionary(uniqueKeysWithValues: pairing.sessions.map { ($0.sessionID, $0.deviceName) })
        var tiles = streaming.feedStates.map { tile in
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

        for index in tiles.indices {
            guard tiles[index].latestDetections.isEmpty, let latestFrame = tiles[index].latestFrame else { continue }
            let detections = await remoteInference.inferDetections(for: latestFrame, sessionID: tiles[index].sessionID)
            guard !detections.isEmpty else { continue }
            tiles[index].latestDetections = detections
            streaming.publishDetections(detections, for: tiles[index].sessionID)
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
