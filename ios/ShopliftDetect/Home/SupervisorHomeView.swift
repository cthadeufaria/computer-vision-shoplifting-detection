import SwiftUI

struct SupervisorHomeView: View {
    var body: some View {
        VStack(spacing: 20) {
            Spacer()
            Image(systemName: "rectangle.3.group.bubble.left")
                .font(.system(size: 72))
                .foregroundStyle(.blue)
            Text("Supervisor Mode")
                .font(.largeTitle.bold())
                .accessibilityIdentifier("supervisorHomeTitle")
            Text("Pair a camera device to begin monitoring live feeds.")
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 24)
            Spacer()
        }
        .padding()
    }
}
