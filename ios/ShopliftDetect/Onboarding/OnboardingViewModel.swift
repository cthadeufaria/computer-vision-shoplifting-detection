import SwiftUI
import AVFoundation

@MainActor
final class OnboardingViewModel: ObservableObject {
    @Published var currentPage = 0
    @Published var selectedRole: DeviceRole?
    @Published var errorMessage: String?

    let totalPages = 4

    private let persistence: PersistenceServiceProtocol
    private let permission: PermissionServiceProtocol

    init(
        persistence: PersistenceServiceProtocol,
        permission: PermissionServiceProtocol
    ) {
        self.persistence = persistence
        self.permission = permission
    }

    var canAdvance: Bool {
        currentPage != 2 || selectedRole != nil
    }

    func nextPage() {
        guard canAdvance else { return }
        currentPage = min(currentPage + 1, totalPages - 1)
    }

    func selectRole(_ role: DeviceRole) {
        selectedRole = role
    }

    func completeAfterPermissions() async {
        errorMessage = nil

        let status = permission.cameraAuthorizationStatus()
        if status == .notDetermined {
            await permission.requestCameraAccess()
        }

        let finalStatus = permission.cameraAuthorizationStatus()
        guard finalStatus != .denied, finalStatus != .restricted else {
            errorMessage = "Camera access is required to finish setup for live detection or QR pairing."
            return
        }

        complete()
    }

    func requestCameraPermission() async {
        await completeAfterPermissions()
    }

    func permissionSummaryText() -> String {
        switch selectedRole {
        case .camera:
            return "Grant camera access so this device can run live pose detection."
        case .supervisor:
            return "Grant camera access so this device can scan pairing QR codes."
        case .none:
            return "Grant camera access to finish setup."
        }
    }

    func selectedRoleTitle() -> String {
        switch selectedRole {
        case .camera:
            return "Smart Camera"
        case .supervisor:
            return "Supervisory View"
        case .none:
            return "Select a role"
        }
    }

    func complete() {
        persistence.selectedRole = selectedRole
        persistence.onboardingComplete = true
    }
}
