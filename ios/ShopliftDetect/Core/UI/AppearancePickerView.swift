import SwiftUI

struct AppearancePickerView: View {
    let selectedAppearance: AppAppearance
    let onSelect: (AppAppearance) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Appearance")
                .font(.headline)

            Picker("Appearance", selection: selectionBinding) {
                ForEach(AppAppearance.allCases, id: \.self) { appearance in
                    Text(appearance.displayName)
                        .tag(appearance)
                }
            }
            .pickerStyle(.segmented)
            .accessibilityIdentifier("appearancePicker")
            .accessibilityValue(selectedAppearance.displayName)
        }
        .padding(16)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    private var selectionBinding: Binding<AppAppearance> {
        Binding(
            get: { selectedAppearance },
            set: onSelect
        )
    }
}
