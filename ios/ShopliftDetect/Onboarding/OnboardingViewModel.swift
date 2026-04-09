import SwiftUI
import AVFoundation

@MainActor
final class OnboardingViewModel: ObservableObject {
    @Published var currentPage = 0
    @Published var selectedRole: DeviceRole?
    @Published var scannedPayload = ""
    @Published var errorMessage: String?
    @Published private(set) var connectionState: ConnectionState = .idle
    @Published private(set) var qrPayload: String?

    let totalPages = 4

    private let persistence: PersistenceServiceProtocol
    private let permission: PermissionServiceProtocol
    private let pairing: PairingServiceProtocol
    private let prefilledSupervisorPayload: String?

    init(
        persistence: PersistenceServiceProtocol,
        permission: PermissionServiceProtocol,
        pairing: PairingServiceProtocol,
        prefilledSupervisorPayload: String? = nil
    ) {
        self.persistence = persistence
        self.permission = permission
        self.pairing = pairing
        self.prefilledSupervisorPayload = prefilledSupervisorPayload
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

        if selectedRole == .supervisor, pairing.connectionState != .connected {
            errorMessage = "Scan a valid camera pairing code before finishing setup."
            return
        }

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

    func updatePairingScreenVisibility(isVisible: Bool) {
        guard currentPage == totalPages - 1 || !isVisible else { return }

        if isVisible {
            preparePairingContentIfNeeded()
        } else if selectedRole == .camera {
            pairing.expireCameraPairing()
            syncPairingState()
        }
    }

    func scanQRCode() {
        errorMessage = nil
        pairing.connectToCamera(using: scannedPayload, deviceName: "Supervisory View")
        syncPairingState()

        if case .failed(let reason) = connectionState {
            errorMessage = errorMessage(for: reason)
        }
    }

    func permissionSummaryText() -> String {
        switch selectedRole {
        case .camera:
            return "Grant camera access so this device can run live pose detection and keep the pairing code visible while you onboard."
        case .supervisor:
            return "Grant camera access so this device can scan pairing QR codes, then connect to a smart camera."
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

    func connectionStateText() -> String {
        switch connectionState {
        case .connected:
            return "Connected"
        case .listening:
            return "Waiting for Supervisor"
        case .connecting, .handshaking:
            return "Connecting"
        case .stale:
            return "Connection Stale"
        case .disconnected:
            return "Disconnected"
        case .failed:
            return "Scan Failed"
        case .idle:
            return selectedRole == .camera ? "Preparing QR Code" : "Ready to Scan"
        }
    }

    func permissionButtonTitle() -> String {
        if permission.cameraAuthorizationStatus() == .authorized {
            return "Finish Setup"
        }
        return "Grant Camera Access"
    }

    private func preparePairingContentIfNeeded() {
        switch selectedRole {
        case .camera:
            qrPayload = pairing.prepareCameraPairing(deviceName: "Smart Camera")
        case .supervisor:
            pairing.prepareSupervisorPairing()
            if scannedPayload.isEmpty, let prefilledSupervisorPayload {
                scannedPayload = prefilledSupervisorPayload
            }
        case .none:
            break
        }

        syncPairingState()
    }

    private func syncPairingState() {
        connectionState = pairing.connectionState
        qrPayload = pairing.qrPayloadString
    }

    private func errorMessage(for reason: String) -> String {
        switch reason {
        case PairingFailureReason.invalidPayload.rawValue:
            return "The QR code is invalid. Scan a camera pairing code on the same Wi-Fi network."
        case PairingFailureReason.invalidToken.rawValue:
            return "The pairing token is invalid. Ask the camera device to show a fresh QR code and rescan."
        case PairingFailureReason.expiredToken.rawValue:
            return "That QR code has expired. Open the pairing screen again on the camera device and rescan."
        case PairingFailureReason.reusedToken.rawValue:
            return "That QR code was already used. Show a fresh QR code on the camera device."
        default:
            return "The pairing attempt failed. Rescan the QR code and try again."
        }
    }
}
