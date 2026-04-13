import Foundation

@MainActor
protocol SettingsServiceProtocol: AnyObject {
    var anomalyThreshold: Float { get set }
    var appAppearance: AppAppearance { get set }
}
