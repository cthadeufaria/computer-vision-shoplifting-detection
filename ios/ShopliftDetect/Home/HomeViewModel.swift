import SwiftUI

enum HomeDestination: Equatable {
    case camera
    case supervisor
}

@MainActor
final class HomeViewModel: ObservableObject {
    @Published var isCameraStreamingActive = false
    @Published var isPosePreviewActive = false

    private let persistence: PersistenceServiceProtocol
    private let settings: SettingsServiceProtocol
    private let pairing: PairingServiceProtocol
    private let capabilities: DeviceCapabilities

    init(
        persistence: PersistenceServiceProtocol,
        settings: SettingsServiceProtocol,
        pairing: PairingServiceProtocol,
        capabilities: DeviceCapabilities
    ) {
        self.persistence = persistence
        self.settings = settings
        self.pairing = pairing
        self.capabilities = capabilities
    }

    var selectedRole: DeviceRole? {
        persistence.selectedRole
    }

    var destination: HomeDestination {
        switch selectedRole {
        case .supervisor:
            return capabilities.supportsSupervisorRole ? .supervisor : .camera
        case .camera, .none:
            return .camera
        }
    }

    var anomalyThreshold: Float {
        settings.anomalyThreshold
    }

    var pairingStatusText: String {
        switch pairing.connectionState {
        case .connected:
            return "Connected"
        case .listening:
            return "Pairing Available"
        case .connecting, .handshaking:
            return "Connecting"
        case .stale:
            return "Connection Stale"
        case .disconnected:
            return "Disconnected"
        case .failed:
            return "Pairing Failed"
        case .idle:
            return "Not Paired"
        }
    }

    var cameraPrimaryActionTitle: String {
        "Start Streaming"
    }

    var cameraModeDescription: String {
        if capabilities.supportsOnDeviceInference {
            return "Stream this camera over local Wi-Fi to a supervisory device for inference."
        }
        return "This device runs as a smart camera only and streams frames over local Wi-Fi."
    }

    var canShowPosePreview: Bool {
        capabilities.supportsPosePreview
    }
}
