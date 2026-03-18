import Foundation

/// Shared environment values passed through the view hierarchy.
@MainActor
final class AppEnvironment: ObservableObject {
    static let shared = AppEnvironment()
    private init() {}
}
