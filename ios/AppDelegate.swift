import UIKit

/// Controls app-wide interface orientation. Set `cameraActive = true` while any
/// camera view is on screen to lock the device to portrait.
@MainActor
final class OrientationLock {
    static let shared = OrientationLock()
    private init() {}
    var cameraActive = false
}

final class AppDelegate: NSObject, UIApplicationDelegate {
    func application(
        _ application: UIApplication,
        supportedInterfaceOrientationsFor window: UIWindow?
    ) -> UIInterfaceOrientationMask {
        OrientationLock.shared.cameraActive ? .portrait : .allButUpsideDown
    }
}
