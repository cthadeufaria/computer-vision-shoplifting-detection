import SwiftUI

enum HomeDestination: Equatable {
    case camera
    case supervisor
}

@MainActor
final class HomeViewModel: ObservableObject {
    @Published var isDetectionActive = false
    @Published var isPosePreviewActive = false

    private let persistence: PersistenceServiceProtocol
    private let settings: SettingsServiceProtocol
    private let pairing: PairingServiceProtocol

    init(
        persistence: PersistenceServiceProtocol,
        settings: SettingsServiceProtocol,
        pairing: PairingServiceProtocol
    ) {
        self.persistence = persistence
        self.settings = settings
        self.pairing = pairing
    }

    var selectedRole: DeviceRole? {
        persistence.selectedRole
    }

    var destination: HomeDestination {
        switch selectedRole {
        case .supervisor:
            return .supervisor
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
}
