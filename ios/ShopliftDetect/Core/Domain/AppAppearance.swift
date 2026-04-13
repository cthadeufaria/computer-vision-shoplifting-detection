import Foundation

enum AppAppearance: String, CaseIterable, Equatable, Sendable {
    case light
    case dark

    var displayName: String {
        switch self {
        case .light:
            return "Light"
        case .dark:
            return "Dark"
        }
    }
}
