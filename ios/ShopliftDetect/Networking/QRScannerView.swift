import SwiftUI

struct QRScannerView: View {
    @Binding var payloadText: String
    let connectionStateText: String
    let onScan: () -> Void

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "qrcode.viewfinder")
                .font(.system(size: 72))
                .foregroundStyle(.blue)

            TextField("Paste or scan pairing payload", text: $payloadText)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .textFieldStyle(.roundedBorder)
                .accessibilityIdentifier("qrPayloadTextField")

            Button("Scan QR Code") {
                onScan()
            }
            .buttonStyle(.borderedProminent)
            .accessibilityIdentifier("scanQRCodeButton")

            Text(connectionStateText)
                .font(.headline)
                .foregroundStyle(.secondary)
                .accessibilityIdentifier("pairingConnectionStatusLabel")
        }
    }
}
