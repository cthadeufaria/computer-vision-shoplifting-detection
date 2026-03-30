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
        sut.complete()
        XCTAssertTrue(mockPersistence.onboardingComplete)
    }

    func test_requestCameraPermission_callsPermissionService() async {
        await sut.requestCameraPermission()
        XCTAssertEqual(mockPermission.requestCallCount, 1)
    }

    func test_requestCameraPermission_setsOnboardingCompleteInPersistence() async {
        await sut.requestCameraPermission()
        XCTAssertTrue(mockPersistence.onboardingComplete)
    }

    func test_init_currentPageIsZero() {
        XCTAssertEqual(sut.currentPage, 0)
    }

    func test_totalPagesIsThree() {
        XCTAssertEqual(sut.totalPages, 3)
    }
}
