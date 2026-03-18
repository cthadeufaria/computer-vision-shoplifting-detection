import CoreML

protocol STGNFModelProtocol {
    func runInference(on input: MLMultiArray) throws -> Float
}

/// Wraps the Xcode-generated STGNFModel class (auto-generated from STGNFModel.mlpackage).
/// anomaly_score = -NLL (more negative = more anomalous).
final class STGNFModelRunner: STGNFModelProtocol {
    // STGNFModel is the Xcode-generated class from STGNFModel.mlpackage.
    private let coremlModel: STGNFModel

    init() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        coremlModel = try STGNFModel(configuration: config)
    }

    func runInference(on input: MLMultiArray) throws -> Float {
        let output = try coremlModel.prediction(pose_window: input)
        let nll = Float(output.nll_score[0].doubleValue)
        return -nll  // anomaly_score = -NLL
    }
}

enum STGNFModelError: Error {
    case modelNotFound
    case outputMissing
}
