import Foundation

struct VideoFrame: Equatable, Sendable {
    let timestamp: UInt64
    let jpegData: Data
    let width: Int
    let height: Int
}
