import SwiftUI

@main
struct ShopliftDetectApp: App {
    @StateObject private var appEnvironment: AppEnvironment

    init() {
        let arguments = CommandLine.arguments
        let permissionService: PermissionServiceProtocol = arguments.contains("--ui-test-camera-authorized")
            ? UITestPermissionService()
            : AVPermissionService()
        let requiredExternalToken = arguments
            .first(where: { $0.hasPrefix("--ui-test-required-token=") })?
            .replacingOccurrences(of: "--ui-test-required-token=", with: "")
        let environment = AppEnvironment(
            permissionService: permissionService,
            pairingService: PairingService(externalValidationToken: requiredExternalToken),
            launchArguments: arguments
        )
        environment.applyLaunchArguments(arguments)
        _appEnvironment = StateObject(wrappedValue: environment)
    }

    var body: some Scene {
        WindowGroup {
            Group {
                if appEnvironment.onboardingComplete {
                    HomeView(viewModel: appEnvironment.makeHomeViewModel())
                } else {
                    OnboardingView(viewModel: appEnvironment.makeOnboardingViewModel())
                }
            }
            .environmentObject(appEnvironment)
        }
    }
}
