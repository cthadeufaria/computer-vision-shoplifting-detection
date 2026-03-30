import SwiftUI
import AVFoundation

struct DetectionView: View {
    @Binding var isPresented: Bool
    @StateObject private var viewModel = DetectionViewModel()
    @ObservedObject private var rotation = DeviceRotation.shared
    @State private var startError: String?
    private let isPreviewUITest = ProcessInfo.processInfo.arguments.contains("--ui-test-detection-preview")

    var body: some View {
        ZStack {
            CameraPreviewLayer(previewLayer: viewModel.previewLayer)
                .ignoresSafeArea()
            SkeletonOverlayView(skeletons: viewModel.skeletons, previewLayer: viewModel.previewLayer)
                .ignoresSafeArea()
            DetectionScoreCardOverlay(state: viewModel.detectionState, rotation: rotation.angle)
            WarmupIndicatorView(state: viewModel.detectionState, rotation: rotation.angle)
            DetectionDismissButton(rotation: rotation.angle) {
                viewModel.stop()
                isPresented = false
            }
        }
        .task {
            if isPreviewUITest {
                viewModel.enablePreviewTestMode()
                return
            }
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
    }
}
