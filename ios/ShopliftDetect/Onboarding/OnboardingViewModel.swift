import SwiftUI

@MainActor
final class OnboardingViewModel: ObservableObject {
    @Published var currentPage = 0
    let totalPages = 3

    private let persistence: PersistenceServiceProtocol
    private let permission: PermissionServiceProtocol

    init(
        persistence: PersistenceServiceProtocol = UserDefaultsPersistenceService(),
        permission: PermissionServiceProtocol = AVPermissionService()
    ) {
        self.persistence = persistence
        self.permission = permission
    }

    func requestCameraPermission() async {
        await permission.requestCameraAccess()
        persistence.onboardingComplete = true
    }

    func complete() {
        persistence.onboardingComplete = true
    }
}
