@testable import ShopliftDetect

final class MockPersistenceService: PersistenceServiceProtocol {
    var onboardingComplete = false
    var selectedRole: DeviceRole?
    var detectionSettings = DetectionSettings.default
    var appAppearance: AppAppearance = .light
}
