import SwiftUI

struct OnboardingView: View {
    @StateObject private var viewModel = OnboardingViewModel()

    var body: some View {
        TabView(selection: $viewModel.currentPage) {
            OnboardingPageView(
                title: "Welcome",
                description: "Privacy-preserving shoplifting detection using pose estimation — no video stored.",
                systemImage: "figure.walk"
            )
            .tag(0)

            OnboardingPageView(
                title: "How It Works",
                description: "Your camera detects body poses in real time. The AI analyses movement patterns to flag anomalies — without recording any video.",
                systemImage: "cpu"
            )
            .tag(1)

            VStack(spacing: 24) {
                OnboardingPageView(
                    title: "Camera Access",
                    description: "Grant camera access so the app can analyse poses locally on your device.",
                    systemImage: "camera.fill"
                )

                Button("Grant Camera Access") {
                    Task { await viewModel.requestCameraPermission() }
                }
                .buttonStyle(.borderedProminent)
                .accessibilityIdentifier("grantCameraAccessButton")
            }
            .tag(2)
        }
        .tabViewStyle(.page(indexDisplayMode: .always))
        .animation(.easeInOut, value: viewModel.currentPage)
        .overlay(alignment: .bottomTrailing) {
            if viewModel.currentPage < 2 {
                Button("Next") {
                    withAnimation { viewModel.currentPage += 1 }
                }
                .buttonStyle(.bordered)
                .padding()
                .accessibilityIdentifier("nextButton")
            }
        }
    }
}
