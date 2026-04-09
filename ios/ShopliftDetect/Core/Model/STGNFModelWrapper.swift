import CoreML

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
    private let coremlModel: MLModel

    init() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all

        guard let modelURL = Bundle.main.url(forResource: "STGNFModel", withExtension: "mlpackage")
            ?? Bundle.main.url(forResource: "STGNFModel", withExtension: "mlmodelc")
            ?? Bundle(for: Self.self).url(forResource: "STGNFModel", withExtension: "mlpackage")
            ?? Bundle(for: Self.self).url(forResource: "STGNFModel", withExtension: "mlmodelc") else {
            throw STGNFModelError.modelNotFound
        }

        coremlModel = try MLModel(contentsOf: modelURL, configuration: config)
    }

    func runInference(on input: MLMultiArray) throws -> Float {
        let provider = STGNFInputProvider(poseWindow: input)
        let output = try coremlModel.prediction(from: provider)

        guard let nllFeature = output.featureValue(for: "nll_score"),
              let nllArray = nllFeature.multiArrayValue else {
            throw STGNFModelError.outputMissing
        }

        let nll = Float(truncating: nllArray[0])
        return -nll
    }
}

enum STGNFModelError: Error {
    case modelNotFound
    case outputMissing
}
