import Foundation

@MainActor
final class NoopStreamingService: StreamingServiceProtocol {
    private(set) var isStreaming = false

    func startStreaming() {
        isStreaming = true
    }

    func stopStreaming() {
        isStreaming = false
    }
}
