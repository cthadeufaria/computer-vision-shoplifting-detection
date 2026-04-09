import Foundation

@MainActor
protocol PersistenceServiceProtocol: AnyObject {
    var onboardingComplete: Bool { get set }
    var selectedRole: DeviceRole? { get set }
}

final class UserDefaultsPersistenceService: PersistenceServiceProtocol {
    private enum Keys {
        static let onboardingComplete = "onboardingComplete"
        static let selectedRole = "selectedRole"
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
}
