import Foundation

@MainActor
final class NoopStreamingService: StreamingServiceProtocol {
    private(set) var isStreaming = false

    func stopStreaming() {
        isStreaming = false
    }
}
