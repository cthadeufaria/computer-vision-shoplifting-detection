import Foundation

@MainActor
protocol PersistenceServiceProtocol: AnyObject {
    var onboardingComplete: Bool { get set }
}

final class UserDefaultsPersistenceService: PersistenceServiceProtocol {
    private let key = "onboardingComplete"

    var onboardingComplete: Bool {
        get { UserDefaults.standard.bool(forKey: key) }
        set { UserDefaults.standard.set(newValue, forKey: key) }
    }
}
