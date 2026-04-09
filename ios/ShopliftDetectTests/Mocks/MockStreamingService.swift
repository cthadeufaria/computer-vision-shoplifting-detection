@testable import ShopliftDetect

@MainActor
final class MockStreamingService: StreamingServiceProtocol {
    var isStreaming = false
    private(set) var stopCallCount = 0

    func stopStreaming() {
        stopCallCount += 1
        isStreaming = false
    }
}
