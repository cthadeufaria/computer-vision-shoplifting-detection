import SwiftUI

struct PosePreviewView: View {
    @Binding var isPresented: Bool
    @StateObject private var viewModel: PosePreviewViewModel
    @ObservedObject private var rotation = DeviceRotation.shared
    @State private var startError: String?
    private let isPreviewUITest = ProcessInfo.processInfo.arguments.contains("--ui-test-pose-preview")

    init(
        isPresented: Binding<Bool>,
        viewModel: @autoclosure @escaping () -> PosePreviewViewModel
    ) {
        _isPresented = isPresented
        _viewModel = StateObject(wrappedValue: viewModel())
    }

    var body: some View {
        ZStack {
            CameraPreviewLayer(previewLayer: viewModel.previewLayer)
                .ignoresSafeArea()
            SkeletonOverlayView(skeletons: viewModel.skeletons, previewLayer: viewModel.previewLayer)
                .ignoresSafeArea()
            PosePreviewTopBar(skeletonCount: viewModel.skeletons.count, rotation: rotation.angle) {
                viewModel.stop()
                isPresented = false
            }
            PoseDebugOverlay(debugInfo: viewModel.debugInfo)
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
