import SwiftUI

struct HomeView: View {
    @EnvironmentObject private var appEnvironment: AppEnvironment
    @StateObject private var viewModel: HomeViewModel

    init(viewModel: @autoclosure @escaping () -> HomeViewModel) {
        _viewModel = StateObject(wrappedValue: viewModel())
    }

    var body: some View {
        NavigationView {
            Group {
                switch viewModel.destination {
                case .camera:
                    cameraHome
                case .supervisor:
                    supervisorHome
                }
            }
            .fullScreenCover(isPresented: $viewModel.isCameraStreamingActive) {
                CameraStreamingView(
                    isPresented: $viewModel.isCameraStreamingActive,
                    viewModel: appEnvironment.makeCameraStreamingViewModel()
                )
            }
            .fullScreenCover(isPresented: $viewModel.isPosePreviewActive) {
                PosePreviewView(
                    isPresented: $viewModel.isPosePreviewActive,
                    viewModel: appEnvironment.makePosePreviewViewModel()
                )
            }
        }
        .navigationViewStyle(.stack)
        .screenAppearanceIdentifier("homeScreen")
    }

    private var cameraHome: some View {
        VStack(spacing: 28) {
            AppearancePickerView(selectedAppearance: viewModel.selectedAppearance) { appearance in
                viewModel.selectAppearance(appearance)
            }
            .padding(.top)
            .padding(.horizontal)

            Spacer()
            Image(systemName: "eye.trianglebadge.exclamationmark")
                .font(.system(size: 80))
                .foregroundStyle(.blue)
            Text("ShopliftDetect")
                .font(.largeTitle.bold())
            Text(viewModel.cameraModeDescription)
                .font(.headline)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
                .padding(.horizontal)
                .accessibilityIdentifier("homeThresholdLabel")
            Text(viewModel.pairingStatusText)
                .font(.headline)
                .foregroundStyle(.secondary)
                .accessibilityIdentifier("homePairingStatusLabel")
            Spacer()
            VStack(spacing: 16) {
                Button(viewModel.cameraPrimaryActionTitle) {
                    viewModel.isCameraStreamingActive = true
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .accessibilityIdentifier("startDetectionButton")

                if viewModel.canShowPosePreview {
                    Button("Pose Preview") {
                        viewModel.isPosePreviewActive = true
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.large)
                    .accessibilityIdentifier("posePreviewButton")
                }
            }
            Spacer()
        }
    }

    private var supervisorHome: some View {
        VStack(spacing: 16) {
            AppearancePickerView(selectedAppearance: viewModel.selectedAppearance) { appearance in
                viewModel.selectAppearance(appearance)
            }
            .padding(.horizontal)
            .padding(.top)

            SupervisorHomeView(
                connectionStatusText: viewModel.pairingStatusText,
                viewModel: appEnvironment.makeSupervisorViewModel()
            )
        }
    }
}
