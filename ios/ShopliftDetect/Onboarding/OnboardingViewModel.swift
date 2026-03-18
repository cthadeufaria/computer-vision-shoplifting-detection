import SwiftUI
import AVFoundation

@MainActor
final class OnboardingViewModel: ObservableObject {
    @Published var currentPage = 0
    let totalPages = 3

    @AppStorage("onboardingComplete") var onboardingComplete = false

    func requestCameraPermission() async {
        await AVCaptureDevice.requestAccess(for: .video)
        onboardingComplete = true
    }

    func complete() {
        onboardingComplete = true
    }
}
