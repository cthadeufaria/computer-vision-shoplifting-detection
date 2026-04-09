import SwiftUI

struct SupervisorHomeView: View {
    let connectionStatusText: String
    @StateObject private var viewModel: SupervisorViewModel

    init(
        connectionStatusText: String,
        viewModel: @autoclosure @escaping () -> SupervisorViewModel
    ) {
        self.connectionStatusText = connectionStatusText
        _viewModel = StateObject(wrappedValue: viewModel())
    }

    var body: some View {
        VStack(spacing: 12) {
            HStack {
                Text("Supervisor Mode")
                    .font(.largeTitle.bold())
                    .accessibilityIdentifier("supervisorHomeTitle")
                Spacer()
                Text(connectionStatusText)
                    .font(.headline)
                    .foregroundStyle(.secondary)
                    .accessibilityIdentifier("supervisorConnectionStatusLabel")
            }

            SupervisorView(viewModel: viewModel)
        }
        .padding()
    }
}
