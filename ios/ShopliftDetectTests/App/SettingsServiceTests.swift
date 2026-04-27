import XCTest
@testable import ShopliftDetect

@MainActor
final class SettingsServiceTests: XCTestCase {
    func test_settingsService_defaultsToLightAppearance() {
        let persistence = MockPersistenceService()
        let sut = UserDefaultsSettingsService(persistence: persistence)

        XCTAssertEqual(sut.appAppearance, .light)
    }

    func test_settingsService_persistsDarkAppearanceThroughPersistenceService() {
        let persistence = MockPersistenceService()
        let sut = UserDefaultsSettingsService(persistence: persistence)

        sut.appAppearance = .dark

        XCTAssertEqual(persistence.appAppearance, .dark)
        XCTAssertEqual(sut.appAppearance, .dark)
    }

    func test_userDefaultsPersistenceService_restoresSavedAppearanceOnNextLaunch() throws {
        let suiteName = "SettingsServiceTests.\(#function)"
        let userDefaults = try XCTUnwrap(UserDefaults(suiteName: suiteName))
        userDefaults.removePersistentDomain(forName: suiteName)

        let firstLaunch = UserDefaultsPersistenceService(userDefaults: userDefaults)
        XCTAssertEqual(firstLaunch.appAppearance, .light)

        firstLaunch.appAppearance = .dark

        let secondLaunch = UserDefaultsPersistenceService(userDefaults: userDefaults)
        XCTAssertEqual(secondLaunch.appAppearance, .dark)

        userDefaults.removePersistentDomain(forName: suiteName)
    }
}
