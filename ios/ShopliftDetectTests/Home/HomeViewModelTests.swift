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
            pairing: MockPairingService(),
            capabilities: MockDeviceCapabilitiesService().currentCapabilities
        )

        XCTAssertEqual(sut.selectedRole, .camera)
        XCTAssertEqual(sut.destination, .camera)
    }

    func test_selectAppearance_updatesSettingsAndPublishesSelection() {
        let settings = MockSettingsService()
        var selectedAppearance: AppAppearance?
        let sut = HomeViewModel(
            persistence: MockPersistenceService(),
            settings: settings,
            pairing: MockPairingService(),
            capabilities: MockDeviceCapabilitiesService().currentCapabilities,
            onAppearanceSelected: { selectedAppearance = $0 }
        )

        sut.selectAppearance(.dark)

        XCTAssertEqual(sut.selectedAppearance, .dark)
        XCTAssertEqual(settings.appAppearance, .dark)
        XCTAssertEqual(selectedAppearance, .dark)
    }

    func test_anomalyThreshold_readsFromSettings() {
        let settings = MockSettingsService()
        settings.anomalyThreshold = -0.8
        let sut = HomeViewModel(
            persistence: MockPersistenceService(),
            settings: settings,
            pairing: MockPairingService(),
            capabilities: MockDeviceCapabilitiesService().currentCapabilities
        )

        XCTAssertEqual(sut.anomalyThreshold, -0.8, accuracy: 0.0001)
    }

    func test_destination_defaultsToCameraWhenRoleMissing() {
        let sut = HomeViewModel(
            persistence: MockPersistenceService(),
            settings: MockSettingsService(),
            pairing: MockPairingService(),
            capabilities: MockDeviceCapabilitiesService().currentCapabilities
        )

        XCTAssertEqual(sut.destination, .camera)
    }

    func test_destination_readsSupervisorRole() {
        let persistence = MockPersistenceService()
        persistence.selectedRole = .supervisor
        let sut = HomeViewModel(
            persistence: persistence,
            settings: MockSettingsService(),
            pairing: MockPairingService(),
            capabilities: MockDeviceCapabilitiesService().currentCapabilities
        )

        XCTAssertEqual(sut.destination, .supervisor)
    }

    func test_pairingStatusText_readsFromPairingService() {
        let pairing = MockPairingService()
        pairing.connectionState = .connected
        let sut = HomeViewModel(
            persistence: MockPersistenceService(),
            settings: MockSettingsService(),
            pairing: pairing,
            capabilities: MockDeviceCapabilitiesService().currentCapabilities
        )

        XCTAssertEqual(sut.pairingStatusText, "Connected")
    }

    func test_destination_fallsBackToCameraWhenSupervisorUnsupported() {
        let persistence = MockPersistenceService()
        persistence.selectedRole = .supervisor
        let sut = HomeViewModel(
            persistence: persistence,
            settings: MockSettingsService(),
            pairing: MockPairingService(),
            capabilities: MockDeviceCapabilitiesService(
                supportsSupervisorRole: false,
                supportsOnDeviceInference: false,
                supportsPosePreview: false
            ).currentCapabilities
        )

        XCTAssertEqual(sut.destination, .camera)
        XCTAssertFalse(sut.canShowPosePreview)
        XCTAssertEqual(sut.cameraPrimaryActionTitle, "Start Streaming")
    }
}
