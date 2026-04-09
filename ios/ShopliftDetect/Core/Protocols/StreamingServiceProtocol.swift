import Foundation

@MainActor
protocol StreamingServiceProtocol: AnyObject {
    var isStreaming: Bool { get }
    func stopStreaming()
}
