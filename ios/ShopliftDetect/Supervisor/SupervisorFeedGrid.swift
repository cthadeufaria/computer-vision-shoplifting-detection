import Foundation

struct SupervisorFeedGrid: Equatable, Sendable {
    static let maxFeeds = 4

    var tiles: [SupervisorFeedTileState]

    init(tiles: [SupervisorFeedTileState] = []) {
        self.tiles = Array(tiles.prefix(Self.maxFeeds))
    }

    var canAddMoreFeeds: Bool {
        tiles.count < Self.maxFeeds
    }
}

