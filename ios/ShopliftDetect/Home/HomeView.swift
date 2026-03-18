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
                Button("Start Detection") {
                    viewModel.isDetectionActive = true
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .accessibilityIdentifier("startDetectionButton")
                Spacer()
            }
            .fullScreenCover(isPresented: $viewModel.isDetectionActive) {
                DetectionView(isPresented: $viewModel.isDetectionActive)
            }
        }
    }
}
