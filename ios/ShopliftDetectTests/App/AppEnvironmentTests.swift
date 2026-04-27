import XCTest
@testable import ShopliftDetect

@MainActor
final class AppEnvironmentTests: XCTestCase {
    func test_init_readsActiveAppearanceFromSettings() {
        let settings = MockSettingsService()
        settings.appAppearance = .dark

        let sut = AppEnvironment(
            persistenceService: MockPersistenceService(),
            permissionService: MockPermissionService(),
            settingsService: settings,
            pairingService: MockPairingService(),
            streamingService: MockStreamingService(),
            deviceCapabilitiesService: MockDeviceCapabilitiesService()
        )

        XCTAssertEqual(sut.activeAppearance, .dark)
    }

    func test_updateAppearance_persistsAndPublishesAppearance() {
        let settings = MockSettingsService()
        let sut = AppEnvironment(
            persistenceService: MockPersistenceService(),
            permissionService: MockPermissionService(),
            settingsService: settings,
            pairingService: MockPairingService(),
            streamingService: MockStreamingService(),
            deviceCapabilitiesService: MockDeviceCapabilitiesService()
        )

        sut.updateAppearance(.dark)

        XCTAssertEqual(settings.appAppearance, .dark)
        XCTAssertEqual(sut.activeAppearance, .dark)
    }

    func test_applyLaunchArguments_darkModeOverridesStoredAppearance() {
        let settings = MockSettingsService()
        settings.appAppearance = .light
        let sut = AppEnvironment(
            persistenceService: MockPersistenceService(),
            permissionService: MockPermissionService(),
            settingsService: settings,
            pairingService: MockPairingService(),
            streamingService: MockStreamingService(),
            deviceCapabilitiesService: MockDeviceCapabilitiesService()
        )

        sut.applyLaunchArguments(["--ui-test-dark-mode"])

        XCTAssertEqual(settings.appAppearance, .dark)
        XCTAssertEqual(sut.activeAppearance, .dark)
    }
}
