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
                    SupervisorHomeView(
                        connectionStatusText: viewModel.pairingStatusText,
                        viewModel: appEnvironment.makeSupervisorViewModel()
                    )
                }
            }
            .fullScreenCover(isPresented: $viewModel.isDetectionActive) {
                DetectionView(
                    isPresented: $viewModel.isDetectionActive,
                    viewModel: appEnvironment.makeDetectionViewModel()
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
    }

    private var cameraHome: some View {
        VStack(spacing: 40) {
            Spacer()
            Image(systemName: "eye.trianglebadge.exclamationmark")
                .font(.system(size: 80))
                .foregroundStyle(.blue)
            Text("ShopliftDetect")
                .font(.largeTitle.bold())
            Text(String(format: "Current threshold %.1f", viewModel.anomalyThreshold))
                .font(.headline.monospacedDigit())
                .foregroundStyle(.secondary)
                .accessibilityIdentifier("homeThresholdLabel")
            Text(viewModel.pairingStatusText)
                .font(.headline)
                .foregroundStyle(.secondary)
                .accessibilityIdentifier("homePairingStatusLabel")
            Spacer()
            VStack(spacing: 16) {
                Button("Start Detection") {
                    viewModel.isDetectionActive = true
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .accessibilityIdentifier("startDetectionButton")

                Button("Pose Preview") {
                    viewModel.isPosePreviewActive = true
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
                .accessibilityIdentifier("posePreviewButton")
            }
            Spacer()
        }
    }
}
