import CoreML

protocol STGNFModelProtocol {
    func runInference(on input: MLMultiArray) throws -> Float
}

/// CoreML wrapper for the STG-NF normalizing flow model.
/// anomaly_score = -NLL (more negative = more anomalous).
final class STGNFModel: STGNFModelProtocol {
    private let model: MLModel

    init() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        // The model is bundled as STGNFModel.mlpackage → compiled to STGNFModel.mlmodelc at build time.
        guard let url = Bundle.main.url(forResource: "STGNFModel", withExtension: "mlmodelc") else {
            throw STGNFModelError.modelNotFound
        }
        model = try MLModel(contentsOf: url, configuration: config)
    }

    func runInference(on input: MLMultiArray) throws -> Float {
        let featureProvider = try MLDictionaryFeatureProvider(dictionary: ["pose_window": input])
        let result = try model.prediction(from: featureProvider)
        guard let nllArray = result.featureValue(for: "nll_score")?.multiArrayValue else {
            throw STGNFModelError.outputMissing
        }
        let nll = nllArray[0].floatValue
        return -nll  // anomaly_score = -NLL
    }
}

enum STGNFModelError: Error {
    case modelNotFound
    case outputMissing
}
