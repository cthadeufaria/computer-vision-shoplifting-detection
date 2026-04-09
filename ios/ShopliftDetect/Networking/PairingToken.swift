import Foundation

struct PairingToken: Equatable, Sendable {
    let value: String
    let issuedAt: Date
    var isConsumed: Bool
    var isVisibleOnScreen: Bool

    init(
        value: String = UUID().uuidString,
        issuedAt: Date = Date(),
        isConsumed: Bool = false,
        isVisibleOnScreen: Bool = true
    ) {
        self.value = value
        self.issuedAt = issuedAt
        self.isConsumed = isConsumed
        self.isVisibleOnScreen = isVisibleOnScreen
    }
}
