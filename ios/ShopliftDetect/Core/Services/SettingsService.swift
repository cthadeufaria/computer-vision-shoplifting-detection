import Foundation

final class UserDefaultsSettingsService: SettingsServiceProtocol {
    private let thresholdKey = "anomalyThreshold"

    var anomalyThreshold: Float {
        get {
            let stored = UserDefaults.standard.object(forKey: thresholdKey)
            guard stored != nil else { return -1.2 }
            return UserDefaults.standard.float(forKey: thresholdKey)
        }
        set {
            UserDefaults.standard.set(newValue, forKey: thresholdKey)
        }
    }
}
