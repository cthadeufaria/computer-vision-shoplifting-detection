import SwiftUI

struct RoleSelectionView: View {
    let selectedRole: DeviceRole?
    let onSelect: (DeviceRole) -> Void

    var body: some View {
        VStack(spacing: 16) {
            Text("Choose This Device's Role")
                .font(.title2.bold())

            Text("Smart Camera runs detection. Supervisory View monitors paired camera feeds.")
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)

            roleButton(
                title: "Smart Camera",
                subtitle: "Use this device to capture video, estimate poses, and detect anomalies.",
                role: .camera,
                accessibilityIdentifier: "cameraRoleButton"
            )

            roleButton(
                title: "Supervisory View",
                subtitle: "Use this device to scan QR codes and monitor paired cameras.",
                role: .supervisor,
                accessibilityIdentifier: "supervisorRoleButton"
            )
        }
        .padding()
    }

    private func roleButton(
        title: String,
        subtitle: String,
        role: DeviceRole,
        accessibilityIdentifier: String
    ) -> some View {
        Button {
            onSelect(role)
        } label: {
            VStack(alignment: .leading, spacing: 6) {
                Text(title)
                    .font(.headline)
                Text(subtitle)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.leading)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding()
            .background(
                RoundedRectangle(cornerRadius: 16)
                    .fill(selectedRole == role ? Color.blue.opacity(0.16) : Color(.secondarySystemBackground))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 16)
                    .stroke(selectedRole == role ? Color.blue : Color.clear, lineWidth: 2)
            )
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier(accessibilityIdentifier)
    }
}
