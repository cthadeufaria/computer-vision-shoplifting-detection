import CoreGraphics
import Foundation

struct StreamProtocol {
    enum StreamError: Error, Equatable {
        case invalidEnvelope
        case unsupportedMessageType(UInt8)
        case invalidPayload
    }

    enum Message: Equatable {
        case heartbeat(timestamp: UInt64)
        case videoFrame(VideoFrame)
        case detectionResults(timestamp: UInt64, results: [DetectionResult])
        case disconnectNotice
    }

    private enum MessageType: UInt8 {
        case heartbeat = 0x01
        case videoFrame = 0x02
        case detectionResults = 0x03
        case disconnectNotice = 0x04
    }

    func encode(_ message: Message) throws -> Data {
        let type: MessageType
        let payload: Data

        switch message {
        case .heartbeat(let timestamp):
            type = .heartbeat
            payload = try JSONEncoder().encode(HeartbeatPayload(timestamp: timestamp))
        case .videoFrame(let frame):
            type = .videoFrame
            payload = encodeVideoFrame(frame)
        case .detectionResults(let timestamp, let results):
            type = .detectionResults
            payload = try JSONEncoder().encode(DetectionResultsPayload(timestamp: timestamp, results: results.map(DetectionResultDTO.init)))
        case .disconnectNotice:
            type = .disconnectNotice
            payload = Data()
        }

        var data = Data([type.rawValue])
        data.appendUInt32(UInt32(payload.count))
        data.append(payload)
        return data
    }

    func decode(_ data: Data) throws -> Message {
        guard data.count >= 5 else {
            throw StreamError.invalidEnvelope
        }

        let typeRaw = data[0]
        guard let type = MessageType(rawValue: typeRaw) else {
            throw StreamError.unsupportedMessageType(typeRaw)
        }

        let payloadLength = Int(data.readUInt32(at: 1))
        let payload = data.subdata(in: 5..<data.count)
        guard payload.count == payloadLength else {
            throw StreamError.invalidEnvelope
        }

        switch type {
        case .heartbeat:
            let heartbeat = try JSONDecoder().decode(HeartbeatPayload.self, from: payload)
            return .heartbeat(timestamp: heartbeat.timestamp)
        case .videoFrame:
            return .videoFrame(try decodeVideoFrame(payload))
        case .detectionResults:
            let resultPayload = try JSONDecoder().decode(DetectionResultsPayload.self, from: payload)
            return .detectionResults(
                timestamp: resultPayload.timestamp,
                results: resultPayload.results.map(\.detectionResult)
            )
        case .disconnectNotice:
            return .disconnectNotice
        }
    }

    private func encodeVideoFrame(_ frame: VideoFrame) -> Data {
        var data = Data()
        data.appendUInt64(frame.timestamp)
        data.appendUInt16(UInt16(frame.width))
        data.appendUInt16(UInt16(frame.height))
        data.appendUInt32(UInt32(frame.jpegData.count))
        data.append(frame.jpegData)
        return data
    }

    private func decodeVideoFrame(_ data: Data) throws -> VideoFrame {
        guard data.count >= 16 else {
            throw StreamError.invalidPayload
        }

        let timestamp = data.readUInt64(at: 0)
        let width = Int(data.readUInt16(at: 8))
        let height = Int(data.readUInt16(at: 10))
        let jpegLength = Int(data.readUInt32(at: 12))
        let jpegData = data.subdata(in: 16..<data.count)

        guard jpegData.count == jpegLength else {
            throw StreamError.invalidPayload
        }

        return VideoFrame(
            timestamp: timestamp,
            jpegData: jpegData,
            width: width,
            height: height
        )
    }
}

private struct HeartbeatPayload: Codable, Equatable {
    let timestamp: UInt64
}

private struct DetectionResultsPayload: Codable, Equatable {
    let timestamp: UInt64
    let results: [DetectionResultDTO]
}

private struct DetectionResultDTO: Codable, Equatable {
    let trackID: Int
    let score: Float
    let label: String
    let keypoints: [KeypointDTO]
    let boundingBox: CGRectDTO
    let timestamp: Double

    init(_ result: DetectionResult) {
        trackID = result.trackID
        score = result.score
        label = DetectionResultDTO.labelString(for: result.label)
        keypoints = result.keypoints.map(KeypointDTO.init)
        boundingBox = CGRectDTO(result.boundingBox)
        timestamp = result.timestamp.timeIntervalSince1970
    }

    var detectionResult: DetectionResult {
        DetectionResult(
            trackID: trackID,
            score: score,
            label: DetectionResultDTO.anomalyLabel(from: label),
            keypoints: keypoints.map(\.keypoint),
            boundingBox: boundingBox.rect,
            timestamp: Date(timeIntervalSince1970: timestamp)
        )
    }

    private static func labelString(for label: AnomalyLabel) -> String {
        switch label {
        case .normal:
            return "normal"
        case .anomaly:
            return "anomaly"
        case .warmup:
            return "warmup"
        }
    }

    private static func anomalyLabel(from value: String) -> AnomalyLabel {
        switch value {
        case "anomaly":
            return .anomaly
        case "warmup":
            return .warmup
        default:
            return .normal
        }
    }
}

private struct KeypointDTO: Codable, Equatable {
    let x: Float
    let y: Float
    let confidence: Float

    init(_ keypoint: Keypoint) {
        x = keypoint.x
        y = keypoint.y
        confidence = keypoint.confidence
    }

    var keypoint: Keypoint {
        Keypoint(x: x, y: y, confidence: confidence)
    }
}

private struct CGRectDTO: Codable, Equatable {
    let x: Double
    let y: Double
    let width: Double
    let height: Double

    init(_ rect: CGRect) {
        x = rect.origin.x
        y = rect.origin.y
        width = rect.size.width
        height = rect.size.height
    }

    var rect: CGRect {
        CGRect(x: x, y: y, width: width, height: height)
    }
}

private extension Data {
    mutating func appendUInt16(_ value: UInt16) {
        var bigEndian = value.bigEndian
        append(Data(bytes: &bigEndian, count: MemoryLayout<UInt16>.size))
    }

    mutating func appendUInt32(_ value: UInt32) {
        var bigEndian = value.bigEndian
        append(Data(bytes: &bigEndian, count: MemoryLayout<UInt32>.size))
    }

    mutating func appendUInt64(_ value: UInt64) {
        var bigEndian = value.bigEndian
        append(Data(bytes: &bigEndian, count: MemoryLayout<UInt64>.size))
    }

    func readUInt16(at offset: Int) -> UInt16 {
        subdata(in: offset..<(offset + 2)).withUnsafeBytes { $0.load(as: UInt16.self).bigEndian }
    }

    func readUInt32(at offset: Int) -> UInt32 {
        subdata(in: offset..<(offset + 4)).withUnsafeBytes { $0.load(as: UInt32.self).bigEndian }
    }

    func readUInt64(at offset: Int) -> UInt64 {
        subdata(in: offset..<(offset + 8)).withUnsafeBytes { $0.load(as: UInt64.self).bigEndian }
    }
}
