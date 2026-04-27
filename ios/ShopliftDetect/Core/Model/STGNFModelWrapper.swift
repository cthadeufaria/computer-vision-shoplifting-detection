import CoreML
import Foundation

protocol STGNFModelProtocol {
    func runInference(on input: MLMultiArray) throws -> Float
}

private final class STGNFInputProvider: MLFeatureProvider {
    let poseWindow: MLMultiArray

    init(poseWindow: MLMultiArray) {
        self.poseWindow = poseWindow
    }

    var featureNames: Set<String> { ["pose_window"] }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        guard featureName == "pose_window" else { return nil }
        return MLFeatureValue(multiArray: poseWindow)
    }
}

/// Wraps the bundled STGNF CoreML package without depending on an iOS 17-only
/// generated model type at compile time.
/// anomaly_score = -NLL (more negative = more anomalous).
final class STGNFModelRunner: STGNFModelProtocol, @unchecked Sendable {
    static let expectedInputShape = [1, 2, 24, 18]
    static let expectedSegmentLength = 24
    static let expectedJointCount = 18
    static let usesConfidenceChannel = false
    static let inputFeatureName = "pose_window"
    static let outputFeatureName = "nll_score"
    private static let packagedModelName = "STGNFModel"
    private static let packagedModelExtension = "modelasset"

    private let coremlModel: MLModel

    init() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all

        let modelURL = try Self.resolveModelURL()

        coremlModel = try MLModel(contentsOf: modelURL, configuration: config)
    }

    func runInference(on input: MLMultiArray) throws -> Float {
        let inputShape = input.shape.map(\.intValue)
        guard inputShape == Self.expectedInputShape else {
            throw STGNFModelError.invalidInputShape(expected: Self.expectedInputShape, actual: inputShape)
        }

        let provider = STGNFInputProvider(poseWindow: input)
        let output = try coremlModel.prediction(from: provider)

        guard let nllFeature = output.featureValue(for: Self.outputFeatureName),
              let nllArray = nllFeature.multiArrayValue else {
            throw STGNFModelError.outputMissing
        }

        let nll = Float(truncating: nllArray[0])
        return -nll
    }

    private static func resolveModelURL() throws -> URL {
        if let compiledURL = findBundledCompiledModelURL() {
            return compiledURL
        }

        guard #available(iOS 17.0, *) else {
            throw STGNFModelError.unsupportedPlatform
        }

        guard let packagedModelURL = findPackagedModelURL() else {
            throw STGNFModelError.modelNotFound
        }

        let stagedPackageURL = try prepareStagedPackage(from: packagedModelURL)
        return try compileStagedPackage(at: stagedPackageURL)
    }

    private static func findBundledCompiledModelURL() -> URL? {
        Bundle.main.url(forResource: packagedModelName, withExtension: "mlmodelc")
            ?? Bundle(for: Self.self).url(forResource: packagedModelName, withExtension: "mlmodelc")
    }

    private static func findPackagedModelURL() -> URL? {
        Bundle.main.url(forResource: packagedModelName, withExtension: packagedModelExtension)
            ?? Bundle(for: Self.self).url(forResource: packagedModelName, withExtension: packagedModelExtension)
    }

    @available(iOS 17.0, *)
    private static func prepareStagedPackage(from packagedModelURL: URL) throws -> URL {
        let fileManager = FileManager.default
        let stagingDirectory = try fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        ).appendingPathComponent("Models", isDirectory: true)

        try fileManager.createDirectory(at: stagingDirectory, withIntermediateDirectories: true)

        let stagedPackageURL = stagingDirectory.appendingPathComponent("\(packagedModelName).mlpackage", isDirectory: true)
        if fileManager.fileExists(atPath: stagedPackageURL.path) {
            try fileManager.removeItem(at: stagedPackageURL)
        }

        try fileManager.copyItem(at: packagedModelURL, to: stagedPackageURL)
        return stagedPackageURL
    }

    @available(iOS 17.0, *)
    private static func compileStagedPackage(at packageURL: URL) throws -> URL {
        let fileManager = FileManager.default
        let compiledContainerDirectory = packageURL.deletingLastPathComponent()
        let compiledModelURL = compiledContainerDirectory.appendingPathComponent("\(packagedModelName).mlmodelc", isDirectory: true)

        if fileManager.fileExists(atPath: compiledModelURL.path) {
            return compiledModelURL
        }

        let temporaryCompiledURL = try MLModel.compileModel(at: packageURL)
        if fileManager.fileExists(atPath: compiledModelURL.path) {
            try fileManager.removeItem(at: compiledModelURL)
        }
        try fileManager.moveItem(at: temporaryCompiledURL, to: compiledModelURL)
        return compiledModelURL
    }
}

enum STGNFModelError: Error {
    case modelNotFound
    case outputMissing
    case invalidInputShape(expected: [Int], actual: [Int])
    case unsupportedPlatform
}
