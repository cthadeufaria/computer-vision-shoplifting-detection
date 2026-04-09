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

    init(
        persistence: PersistenceServiceProtocol,
        settings: SettingsServiceProtocol
    ) {
        self.persistence = persistence
        self.settings = settings
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
}
