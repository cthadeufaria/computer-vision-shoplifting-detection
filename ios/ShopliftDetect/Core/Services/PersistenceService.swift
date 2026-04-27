import Foundation

@MainActor
protocol PersistenceServiceProtocol: AnyObject {
    var onboardingComplete: Bool { get set }
    var selectedRole: DeviceRole? { get set }
    var detectionSettings: DetectionSettings { get set }
    var appAppearance: AppAppearance { get set }
}

final class UserDefaultsPersistenceService: PersistenceServiceProtocol {
    private let userDefaults: UserDefaults

    private enum Keys {
        static let onboardingComplete = "onboardingComplete"
        static let selectedRole = "selectedRole"
        static let anomalyThreshold = "anomalyThreshold"
        static let appAppearance = "appAppearance"
    }

    init(userDefaults: UserDefaults = .standard) {
        self.userDefaults = userDefaults
    }

    var onboardingComplete: Bool {
        get { userDefaults.bool(forKey: Keys.onboardingComplete) }
        set { userDefaults.set(newValue, forKey: Keys.onboardingComplete) }
    }

    var selectedRole: DeviceRole? {
        get {
            guard let rawValue = userDefaults.string(forKey: Keys.selectedRole) else {
                return nil
            }

            return DeviceRole(rawValue: rawValue)
        }
        set {
            userDefaults.set(newValue?.rawValue, forKey: Keys.selectedRole)
        }
    }

    var detectionSettings: DetectionSettings {
        get {
            let threshold = userDefaults.object(forKey: Keys.anomalyThreshold) == nil
                ? DetectionSettings.default.anomalyThreshold
                : userDefaults.float(forKey: Keys.anomalyThreshold)
            return DetectionSettings(anomalyThreshold: threshold)
        }
        set {
            userDefaults.set(newValue.anomalyThreshold, forKey: Keys.anomalyThreshold)
        }
    }

    var appAppearance: AppAppearance {
        get {
            guard let rawValue = userDefaults.string(forKey: Keys.appAppearance),
                  let appearance = AppAppearance(rawValue: rawValue) else {
                return .light
            }

            return appearance
        }
        set {
            userDefaults.set(newValue.rawValue, forKey: Keys.appAppearance)
        }
    }
}
