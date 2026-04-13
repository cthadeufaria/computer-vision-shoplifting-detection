import Foundation
@testable import ShopliftDetect

struct MockDeviceCapabilitiesService: DeviceCapabilitiesServiceProtocol {
    var currentCapabilities: DeviceCapabilities

    init(
        supportsCameraRole: Bool = true,
        supportsSupervisorRole: Bool = true,
        supportsOnDeviceInference: Bool = true,
        supportsPosePreview: Bool = true
    ) {
        currentCapabilities = DeviceCapabilities(
            supportsCameraRole: supportsCameraRole,
            supportsSupervisorRole: supportsSupervisorRole,
            supportsOnDeviceInference: supportsOnDeviceInference,
            supportsPosePreview: supportsPosePreview
        )
    }

    var supportsCameraRole: Bool { currentCapabilities.supportsCameraRole }
    var supportsSupervisorRole: Bool { currentCapabilities.supportsSupervisorRole }
    var supportsOnDeviceInference: Bool { currentCapabilities.supportsOnDeviceInference }
    var supportsPosePreview: Bool { currentCapabilities.supportsPosePreview }
}
