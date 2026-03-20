import SwiftUI

struct HomeView: View {
    @StateObject private var viewModel = HomeViewModel()

    var body: some View {
        NavigationStack {
            VStack(spacing: 40) {
                Spacer()
                Image(systemName: "eye.trianglebadge.exclamationmark")
                    .font(.system(size: 80))
                    .foregroundStyle(.blue)
                Text("ShopliftDetect")
                    .font(.largeTitle.bold())
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
            .fullScreenCover(isPresented: $viewModel.isDetectionActive) {
                DetectionView(isPresented: $viewModel.isDetectionActive)
            }
            .fullScreenCover(isPresented: $viewModel.isPosePreviewActive) {
                PosePreviewView(isPresented: $viewModel.isPosePreviewActive)
            }
        }
    }
}
