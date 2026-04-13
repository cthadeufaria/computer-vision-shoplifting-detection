import XCTest
@testable import ShopliftDetect

@MainActor
final class OnboardingViewModelTests: XCTestCase {
    var sut: OnboardingViewModel!
    var mockPersistence: MockPersistenceService!
    var mockPermission: MockPermissionService!
    var mockPairing: MockPairingService!

    override func setUp() {
        mockPersistence = MockPersistenceService()
        mockPermission = MockPermissionService()
        mockPairing = MockPairingService()
        sut = OnboardingViewModel(
            persistence: mockPersistence,
            permission: mockPermission,
            pairing: mockPairing,
            capabilities: MockDeviceCapabilitiesService().currentCapabilities
        )
    }

    func test_complete_setsOnboardingCompleteInPersistence() {
        sut.selectRole(.camera)
        sut.complete()
        XCTAssertTrue(mockPersistence.onboardingComplete)
        XCTAssertEqual(mockPersistence.selectedRole, .camera)
    }

    func test_complete_whenPermissionUndetermined_requestsCameraAccess() async {
        mockPermission.authorizationStatus = .notDetermined
        sut.selectRole(.camera)

        await sut.completeAfterPermissions()

        XCTAssertEqual(mockPermission.requestCallCount, 1)
    }

    func test_completeAfterPermissions_setsOnboardingCompleteInPersistence() async {
        mockPermission.authorizationStatus = .authorized
        mockPairing.connectionState = .connected
        sut.selectRole(.supervisor)

        await sut.completeAfterPermissions()

        XCTAssertTrue(mockPersistence.onboardingComplete)
        XCTAssertEqual(mockPersistence.selectedRole, .supervisor)
    }

    func test_init_currentPageIsZero() {
        XCTAssertEqual(sut.currentPage, 0)
    }

    func test_totalPagesIsFour() {
        XCTAssertEqual(sut.totalPages, 4)
    }

    func test_canAdvancePastRoleSelection_requiresRole() {
        sut.currentPage = 2

        XCTAssertFalse(sut.canAdvance)

        sut.selectRole(.camera)

        XCTAssertTrue(sut.canAdvance)
    }

    func test_completeAfterPermissions_whenPermissionDenied_setsErrorMessage() async {
        mockPermission.authorizationStatus = .denied
        sut.selectRole(.camera)

        await sut.completeAfterPermissions()

        XCTAssertFalse(mockPersistence.onboardingComplete)
        XCTAssertNotNil(sut.errorMessage)
    }

    func test_nextPage_advancesWhenAllowed() {
        sut.nextPage()
        XCTAssertEqual(sut.currentPage, 1)
    }

    func test_nextPage_doesNotAdvancePastRoleSelectionWithoutRole() {
        sut.currentPage = 2

        sut.nextPage()

        XCTAssertEqual(sut.currentPage, 2)
    }

    func test_updatePairingScreenVisibility_forCameraPreparesQRCode() {
        sut.selectRole(.camera)
        sut.currentPage = sut.totalPages - 1

        sut.updatePairingScreenVisibility(isVisible: true)

        XCTAssertEqual(mockPairing.prepareCameraPairingCallCount, 1)
        XCTAssertEqual(sut.qrPayload, "sdlink://192.168.1.24:7890?token=TEST1234")
        XCTAssertEqual(sut.connectionState, .listening)
    }

    func test_scanQRCode_whenPairingFails_setsErrorMessage() {
        let failingPairing = PairingService(externalValidationToken: "VALID123")
        sut = OnboardingViewModel(
            persistence: mockPersistence,
            permission: mockPermission,
            pairing: failingPairing,
            capabilities: MockDeviceCapabilitiesService().currentCapabilities,
            prefilledSupervisorPayload: "sdlink://192.168.1.24:7890?token=WRONG999"
        )
        sut.selectRole(.supervisor)
        sut.currentPage = sut.totalPages - 1
        sut.updatePairingScreenVisibility(isVisible: true)

        sut.scanQRCode()

        XCTAssertEqual(sut.connectionState, .failed(PairingFailureReason.invalidToken.rawValue))
        XCTAssertEqual(
            sut.errorMessage,
            "The pairing token is invalid. Ask the camera device to show a fresh QR code and rescan."
        )
    }

    func test_selectRole_ignoresUnsupportedSupervisorRole() {
        sut = OnboardingViewModel(
            persistence: mockPersistence,
            permission: mockPermission,
            pairing: mockPairing,
            capabilities: MockDeviceCapabilitiesService(
                supportsSupervisorRole: false,
                supportsOnDeviceInference: false,
                supportsPosePreview: false
            ).currentCapabilities
        )

        sut.selectRole(.supervisor)

        XCTAssertNil(sut.selectedRole)
        XCTAssertEqual(
            sut.supervisorAvailabilityNote,
            "Supervisory View requires a newer iPhone or iPad. This device can run as a Smart Camera."
        )
    }
}
