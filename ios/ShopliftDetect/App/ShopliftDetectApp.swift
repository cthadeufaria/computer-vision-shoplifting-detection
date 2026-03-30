import SwiftUI

@main
struct ShopliftDetectApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @AppStorage("onboardingComplete") private var onboardingComplete = false

    init() {
        let args = CommandLine.arguments
        if args.contains("--reset-onboarding") {
            UserDefaults.standard.set(false, forKey: "onboardingComplete")
        } else if args.contains("--skip-onboarding") {
            UserDefaults.standard.set(true, forKey: "onboardingComplete")
        }
    }

    var body: some Scene {
        WindowGroup {
            if onboardingComplete {
                HomeView()
            } else {
                OnboardingView()
            }
        }
    }
}
