@testable import ShopliftDetect

@MainActor
final class MockStreamingService: StreamingServiceProtocol {
    var isStreaming = false
    private(set) var startCallCount = 0
    private(set) var stopCallCount = 0

    func startStreaming() {
        startCallCount += 1
        isStreaming = true
    }

    func stopStreaming() {
        stopCallCount += 1
        isStreaming = false
    }
}
