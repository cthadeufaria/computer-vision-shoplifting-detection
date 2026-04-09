import Foundation

@MainActor
protocol PersistenceServiceProtocol: AnyObject {
    var onboardingComplete: Bool { get set }
    var selectedRole: DeviceRole? { get set }
    var detectionSettings: DetectionSettings { get set }
}

final class UserDefaultsPersistenceService: PersistenceServiceProtocol {
    private enum Keys {
        static let onboardingComplete = "onboardingComplete"
        static let selectedRole = "selectedRole"
        static let anomalyThreshold = "anomalyThreshold"
    }

    var onboardingComplete: Bool {
        get { UserDefaults.standard.bool(forKey: Keys.onboardingComplete) }
        set { UserDefaults.standard.set(newValue, forKey: Keys.onboardingComplete) }
    }

    var selectedRole: DeviceRole? {
        get {
            guard let rawValue = UserDefaults.standard.string(forKey: Keys.selectedRole) else {
                return nil
            }

            return DeviceRole(rawValue: rawValue)
        }
        set {
            UserDefaults.standard.set(newValue?.rawValue, forKey: Keys.selectedRole)
        }
    }

    var detectionSettings: DetectionSettings {
        get {
            let threshold = UserDefaults.standard.object(forKey: Keys.anomalyThreshold) == nil
                ? DetectionSettings.default.anomalyThreshold
                : UserDefaults.standard.float(forKey: Keys.anomalyThreshold)
            return DetectionSettings(anomalyThreshold: threshold)
        }
        set {
            UserDefaults.standard.set(newValue.anomalyThreshold, forKey: Keys.anomalyThreshold)
        }
    }
}
