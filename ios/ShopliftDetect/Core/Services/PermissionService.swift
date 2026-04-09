import AVFoundation

@MainActor
protocol PermissionServiceProtocol: AnyObject {
    func cameraAuthorizationStatus() -> AVAuthorizationStatus
    func requestCameraAccess() async
}

final class AVPermissionService: PermissionServiceProtocol {
    func cameraAuthorizationStatus() -> AVAuthorizationStatus {
        AVCaptureDevice.authorizationStatus(for: .video)
    }

    func requestCameraAccess() async {
        await AVCaptureDevice.requestAccess(for: .video)
    }
}

final class UITestPermissionService: PermissionServiceProtocol {
    func cameraAuthorizationStatus() -> AVAuthorizationStatus {
        .authorized
    }

    func requestCameraAccess() async {}
}
