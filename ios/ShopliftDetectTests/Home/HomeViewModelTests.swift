import XCTest
@testable import ShopliftDetect

@MainActor
final class HomeViewModelTests: XCTestCase {
    func test_selectedRole_readsFromPersistence() {
        let persistence = MockPersistenceService()
        persistence.selectedRole = .camera
        let sut = HomeViewModel(
            persistence: persistence,
            settings: MockSettingsService(),
            pairing: MockPairingService()
        )

        XCTAssertEqual(sut.selectedRole, .camera)
        XCTAssertEqual(sut.destination, .camera)
    }

    func test_anomalyThreshold_readsFromSettings() {
        let settings = MockSettingsService()
        settings.anomalyThreshold = -0.8
        let sut = HomeViewModel(
            persistence: MockPersistenceService(),
            settings: settings,
            pairing: MockPairingService()
        )

        XCTAssertEqual(sut.anomalyThreshold, -0.8, accuracy: 0.0001)
    }

    func test_destination_defaultsToCameraWhenRoleMissing() {
        let sut = HomeViewModel(
            persistence: MockPersistenceService(),
            settings: MockSettingsService(),
            pairing: MockPairingService()
        )

        XCTAssertEqual(sut.destination, .camera)
    }

    func test_destination_readsSupervisorRole() {
        let persistence = MockPersistenceService()
        persistence.selectedRole = .supervisor
        let sut = HomeViewModel(
            persistence: persistence,
            settings: MockSettingsService(),
            pairing: MockPairingService()
        )

        XCTAssertEqual(sut.destination, .supervisor)
    }

    func test_pairingStatusText_readsFromPairingService() {
        let pairing = MockPairingService()
        pairing.connectionState = .connected
        let sut = HomeViewModel(
            persistence: MockPersistenceService(),
            settings: MockSettingsService(),
            pairing: pairing
        )

        XCTAssertEqual(sut.pairingStatusText, "Connected")
    }
}
