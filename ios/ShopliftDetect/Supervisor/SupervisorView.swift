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
                        .background(Color.orange.opacity(0.14), in: RoundedRectangle(cornerRadius: 12))
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
            viewModel.refresh()
        }
        .refreshable {
            viewModel.refresh()
        }
        .fullScreenCover(item: Binding(
            get: { viewModel.selectedTile },
            set: { if $0 == nil { viewModel.clearSelection() } }
        )) { tile in
            CameraFeedDetailView(tile: tile) {
                viewModel.clearSelection()
            }
        }
    }
}
