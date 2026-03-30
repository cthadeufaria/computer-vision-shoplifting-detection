@testable import ShopliftDetect

final class MockPermissionService: PermissionServiceProtocol {
    var requestCallCount = 0

    func requestCameraAccess() async {
        requestCallCount += 1
    }
}
