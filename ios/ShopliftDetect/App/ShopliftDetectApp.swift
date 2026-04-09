import SwiftUI

@main
struct ShopliftDetectApp: App {
    @StateObject private var appEnvironment: AppEnvironment

    init() {
        let arguments = CommandLine.arguments
        let permissionService: PermissionServiceProtocol = arguments.contains("--ui-test-camera-authorized")
            ? UITestPermissionService()
            : AVPermissionService()
        let environment = AppEnvironment(permissionService: permissionService)
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
