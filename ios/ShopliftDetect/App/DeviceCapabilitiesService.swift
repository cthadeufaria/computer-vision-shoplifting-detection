import Foundation

protocol DeviceCapabilitiesServiceProtocol {
    var currentCapabilities: DeviceCapabilities { get }
    var supportsCameraRole: Bool { get }
    var supportsSupervisorRole: Bool { get }
    var supportsOnDeviceInference: Bool { get }
    var supportsPosePreview: Bool { get }
}

struct DeviceCapabilities: Equatable, Sendable {
    let supportsCameraRole: Bool
    let supportsSupervisorRole: Bool
    let supportsOnDeviceInference: Bool
    let supportsPosePreview: Bool
}

struct CurrentDeviceCapabilitiesService: DeviceCapabilitiesServiceProtocol {
    let currentCapabilities: DeviceCapabilities

    init(processInfo: ProcessInfo = .processInfo) {
        self.init(launchArguments: processInfo.arguments, operatingSystemVersion: processInfo.operatingSystemVersion)
    }

    init(launchArguments: [String], operatingSystemVersion: OperatingSystemVersion = ProcessInfo.processInfo.operatingSystemVersion) {
        let arguments = launchArguments
        if arguments.contains("--ui-test-camera-only-device") {
            currentCapabilities = DeviceCapabilities(
                supportsCameraRole: true,
                supportsSupervisorRole: false,
                supportsOnDeviceInference: false,
                supportsPosePreview: false
            )
            return
        }

        if arguments.contains("--ui-test-supervisor-capable-device") {
            currentCapabilities = DeviceCapabilities(
                supportsCameraRole: true,
                supportsSupervisorRole: true,
                supportsOnDeviceInference: true,
                supportsPosePreview: true
            )
            return
        }

        let supportsModernRole = operatingSystemVersion.majorVersion >= 17
        currentCapabilities = DeviceCapabilities(
            supportsCameraRole: true,
            supportsSupervisorRole: supportsModernRole,
            supportsOnDeviceInference: supportsModernRole,
            supportsPosePreview: supportsModernRole
        )
    }

    var supportsCameraRole: Bool { currentCapabilities.supportsCameraRole }
    var supportsSupervisorRole: Bool { currentCapabilities.supportsSupervisorRole }
    var supportsOnDeviceInference: Bool { currentCapabilities.supportsOnDeviceInference }
    var supportsPosePreview: Bool { currentCapabilities.supportsPosePreview }
}
