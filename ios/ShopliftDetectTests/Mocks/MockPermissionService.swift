import AVFoundation
@testable import ShopliftDetect

final class MockPermissionService: PermissionServiceProtocol {
    var requestCallCount = 0
    var authorizationStatus: AVAuthorizationStatus = .notDetermined

    func cameraAuthorizationStatus() -> AVAuthorizationStatus {
        authorizationStatus
    }

    func requestCameraAccess() async {
        requestCallCount += 1
    }
}
