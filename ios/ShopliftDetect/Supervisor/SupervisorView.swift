import SwiftUI

struct SupervisorView: View {
    @StateObject private var viewModel: SupervisorViewModel

    init(viewModel: @autoclosure @escaping () -> SupervisorViewModel) {
        _viewModel = StateObject(wrappedValue: viewModel())
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if viewModel.showsLimitBanner {
                    Text("v1 supports up to four simultaneous feeds.")
                        .font(.headline)
                        .padding(12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 12))
                        .overlay {
                            RoundedRectangle(cornerRadius: 12)
                                .stroke(Color.orange.opacity(0.45), lineWidth: 1)
                        }
                        .accessibilityIdentifier("supervisorLimitBanner")
                }

                if viewModel.grid.tiles.isEmpty {
                    Text("Pair a camera device to begin monitoring live feeds.")
                        .font(.body)
                        .foregroundStyle(.secondary)
                        .accessibilityIdentifier("supervisorEmptyStateLabel")
                } else {
                    LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 16) {
                        ForEach(viewModel.grid.tiles) { tile in
                            DeviceRowView(tile: tile) {
                                viewModel.select(tile)
                            }
                        }
                    }
                }
            }
            .padding()
        }
        .navigationTitle("Supervisor Mode")
        .task {
            await viewModel.refresh()
        }
        .refreshable {
            await viewModel.refresh()
        }
        .fullScreenCover(item: Binding(
            get: { viewModel.selectedTile },
            set: { if $0 == nil { viewModel.clearSelection() } }
        )) { tile in
            CameraFeedDetailView(tile: tile) {
                viewModel.clearSelection()
            }
        }
        .screenAppearanceIdentifier("supervisorScreen")
    }
}
