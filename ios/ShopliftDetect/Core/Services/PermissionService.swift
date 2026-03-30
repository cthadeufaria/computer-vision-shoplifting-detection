import AVFoundation

@MainActor
protocol PermissionServiceProtocol: AnyObject {
    func requestCameraAccess() async
}

final class AVPermissionService: PermissionServiceProtocol {
    func requestCameraAccess() async {
        await AVCaptureDevice.requestAccess(for: .video)
    }
}
