import SwiftUI

struct QRCodeDisplayView: View {
    let payload: String?
    let connectionStateText: String

    var body: some View {
        VStack(spacing: 16) {
            ZStack {
                RoundedRectangle(cornerRadius: 24)
                    .fill(Color(.secondarySystemBackground))
                    .frame(width: 220, height: 220)

                Image(systemName: "qrcode")
                    .font(.system(size: 112))
                    .foregroundStyle(.blue)
            }
            .accessibilityIdentifier("pairingQRCodeView")

            Text(payload ?? "Preparing pairing code...")
                .font(.footnote.monospaced())
                .multilineTextAlignment(.center)
                .textSelection(.enabled)
                .accessibilityIdentifier("pairingQRCodePayloadLabel")

            Text(connectionStateText)
                .font(.headline)
                .foregroundStyle(.secondary)
                .accessibilityIdentifier("pairingConnectionStatusLabel")
        }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("pairingQRCodeView")
    }
}
