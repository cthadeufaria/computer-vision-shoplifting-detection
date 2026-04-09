import XCTest
@testable import ShopliftDetect

@MainActor
final class OnboardingViewModelTests: XCTestCase {
    var sut: OnboardingViewModel!
    var mockPersistence: MockPersistenceService!
    var mockPermission: MockPermissionService!

    override func setUp() {
        mockPersistence = MockPersistenceService()
        mockPermission = MockPermissionService()
        sut = OnboardingViewModel(
            persistence: mockPersistence,
            permission: mockPermission
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
}
