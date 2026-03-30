import UIKit

/// Controls app-wide interface orientation. Set `cameraActive = true` while any
/// camera view is on screen to lock the device to portrait.
@MainActor
final class OrientationLock {
    static let shared = OrientationLock()
    private init() {}
    var cameraActive = false {
        didSet {
            // Force UIKit to re-query supportedInterfaceOrientationsFor on the next cycle.
            UIApplication.shared.connectedScenes
                .compactMap { $0 as? UIWindowScene }
                .compactMap { $0.keyWindow?.rootViewController }
                .forEach { $0.setNeedsUpdateOfSupportedInterfaceOrientations() }
        }
    }
}

final class AppDelegate: NSObject, UIApplicationDelegate {
    func application(
        _ application: UIApplication,
        supportedInterfaceOrientationsFor window: UIWindow?
    ) -> UIInterfaceOrientationMask {
        OrientationLock.shared.cameraActive ? .portrait : .allButUpsideDown
    }
}
