import SwiftUI

private struct ScreenAppearanceModifier: ViewModifier {
    @Environment(\.colorScheme) private var colorScheme

    let identifier: String

    func body(content: Content) -> some View {
        content.overlay(alignment: .topLeading) {
            Color.clear
                .frame(width: 1, height: 1)
                .accessibilityIdentifier(identifier)
                .accessibilityValue(colorScheme == .dark ? "dark" : "light")
        }
    }
}

extension View {
    func screenAppearanceIdentifier(_ identifier: String) -> some View {
        modifier(ScreenAppearanceModifier(identifier: identifier))
    }
}
