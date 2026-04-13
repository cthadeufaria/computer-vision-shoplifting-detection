@testable import ShopliftDetect

final class MockSettingsService: SettingsServiceProtocol {
    var anomalyThreshold: Float = -1.2
    var appAppearance: AppAppearance = .light
}
