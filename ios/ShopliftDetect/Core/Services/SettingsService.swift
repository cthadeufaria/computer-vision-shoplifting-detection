import Foundation

final class UserDefaultsSettingsService: SettingsServiceProtocol {
    private let persistence: PersistenceServiceProtocol

    init(persistence: PersistenceServiceProtocol = UserDefaultsPersistenceService()) {
        self.persistence = persistence
    }

    var anomalyThreshold: Float {
        get { persistence.detectionSettings.anomalyThreshold }
        set { persistence.detectionSettings = DetectionSettings(anomalyThreshold: newValue) }
    }

    var appAppearance: AppAppearance {
        get { persistence.appAppearance }
        set { persistence.appAppearance = newValue }
    }
}
