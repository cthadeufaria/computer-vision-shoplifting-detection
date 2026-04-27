import SwiftUI

struct CameraStreamingView: View {
    @Binding var isPresented: Bool
    @StateObject private var viewModel: CameraStreamingViewModel
    @State private var startError: String?

    init(
        isPresented: Binding<Bool>,
        viewModel: @autoclosure @escaping () -> CameraStreamingViewModel
    ) {
        _isPresented = isPresented
        _viewModel = StateObject(wrappedValue: viewModel())
    }

    var body: some View {
        ZStack(alignment: .topTrailing) {
            CameraPreviewLayer(previewLayer: viewModel.previewLayer)
                .ignoresSafeArea()

            VStack(spacing: 12) {
                Text(viewModel.statusText)
                    .font(.headline)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 10)
                    .background(.ultraThinMaterial, in: Capsule())
                    .accessibilityIdentifier("streamingStatusLabel")

                Button("Done") {
                    viewModel.stop()
                    isPresented = false
                }
                .buttonStyle(.borderedProminent)
                .padding(.top, 8)
                .accessibilityIdentifier("dismissStreamingButton")
            }
            .padding()
        }
        .task {
            do {
                try viewModel.start()
            } catch {
                startError = error.localizedDescription
            }
        }
        .alert("Camera Error", isPresented: Binding(
            get: { startError != nil },
            set: { if !$0 { startError = nil; isPresented = false } }
        )) {
            Button("OK") { isPresented = false }
        } message: {
            Text(startError ?? "")
        }
        .onDisappear {
            viewModel.stop()
        }
        .screenAppearanceIdentifier("cameraStreamingScreen")
    }
}
