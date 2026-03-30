import SwiftUI
import AVFoundation

struct DetectionView: View {
    @Binding var isPresented: Bool
    @StateObject private var viewModel = DetectionViewModel()
    @State private var startError: String?
    private let isPreviewUITest = ProcessInfo.processInfo.arguments.contains("--ui-test-detection-preview")

    var body: some View {
        ZStack {
            // Layer 1: camera preview
            CameraPreviewLayer(previewLayer: viewModel.previewLayer)
                .ignoresSafeArea()

            // Layer 2: skeleton overlay
            SkeletonOverlayView(skeletons: viewModel.skeletons, previewLayer: viewModel.previewLayer)
                .ignoresSafeArea()

            // Layer 3: score card (top-right)
            VStack {
                HStack {
                    Spacer()
                    ScoreCardView(state: viewModel.detectionState)
                        .padding()
                }
                Spacer()
            }

            // Layer 4: warmup indicator (centered)
            if case .warmingUp(let collected, let needed) = viewModel.detectionState {
                Text("Collecting frames \(collected)/\(needed)")
                    .font(.headline)
                    .padding()
                    .background(.ultraThinMaterial)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .accessibilityIdentifier("warmupIndicator")
            }

            // Layer 5: dismiss button (top-left)
            VStack {
                HStack {
                    Button {
                        viewModel.stop()
                        isPresented = false
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.title)
                            .foregroundStyle(.white)
                    }
                    .accessibilityIdentifier("xmark.circle.fill")
                    .padding()
                    Spacer()
                }
                Spacer()
            }
        }
        .onAppear {
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
