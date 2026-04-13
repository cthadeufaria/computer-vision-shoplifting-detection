import SwiftUI

struct RoleSelectionView: View {
    let selectedRole: DeviceRole?
    let cameraSubtitle: String
    let supervisorSubtitle: String
    let supportsSupervisorRole: Bool
    let supervisorAvailabilityNote: String?
    let onSelect: (DeviceRole) -> Void

    var body: some View {
        VStack(spacing: 16) {
            Text("Choose This Device's Role")
                .font(.title2.bold())

            Text("Smart Camera captures and streams. Supervisory View monitors paired camera feeds and runs inference.")
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)

            roleButton(
                title: "Smart Camera",
                subtitle: cameraSubtitle,
                role: .camera,
                accessibilityIdentifier: "cameraRoleButton",
                isEnabled: true
            )

            roleButton(
                title: "Supervisory View",
                subtitle: supervisorSubtitle,
                role: .supervisor,
                accessibilityIdentifier: "supervisorRoleButton",
                isEnabled: supportsSupervisorRole
            )

            if let supervisorAvailabilityNote {
                Text(supervisorAvailabilityNote)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
                    .accessibilityIdentifier("supervisorAvailabilityNote")
            }
        }
        .padding()
    }

    private func roleButton(
        title: String,
        subtitle: String,
        role: DeviceRole,
        accessibilityIdentifier: String,
        isEnabled: Bool
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
            .opacity(isEnabled ? 1 : 0.5)
        }
        .buttonStyle(.plain)
        .disabled(!isEnabled)
        .accessibilityIdentifier(accessibilityIdentifier)
    }
}
