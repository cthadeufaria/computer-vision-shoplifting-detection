import SwiftUI

struct PosePreviewView: View {
    @Binding var isPresented: Bool
    @StateObject private var viewModel = PosePreviewViewModel()
    @State private var startError: String?

    var body: some View {
        ZStack {
            CameraPreviewLayer(previewLayer: viewModel.previewLayer)
                .ignoresSafeArea()

            SkeletonOverlayView(skeletons: viewModel.skeletons, previewLayer: viewModel.previewLayer)
                .ignoresSafeArea()

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
                    .accessibilityIdentifier("posePreviewDismissButton")
                    .padding()

                    Spacer()

                    Text("Poses: \(viewModel.skeletons.count)")
                        .font(.caption.bold())
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(.ultraThinMaterial)
                        .clipShape(Capsule())
                        .accessibilityIdentifier("posePreviewCount")
                        .padding()
                }

                Spacer()

                // Debug overlay — remove once pose alignment is verified.
                if !viewModel.debugInfo.isEmpty {
                    Text(viewModel.debugInfo)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(.white)
                        .padding(8)
                        .background(Color.black.opacity(0.65))
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                        .padding(.horizontal, 12)
                        .padding(.bottom, 12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
        .onAppear {
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
        .supportedInterfaceOrientations(.portrait)
    }
}
