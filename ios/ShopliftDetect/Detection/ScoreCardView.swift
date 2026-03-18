import SwiftUI

struct ScoreCardView: View {
    let state: DetectionState

    var body: some View {
        VStack(alignment: .trailing, spacing: 4) {
            switch state {
            case .idle:
                EmptyView()
            case .warmingUp(let collected, let needed):
                Text("Warming up \(collected)/\(needed)")
                    .font(.caption.bold())
                    .padding(8)
                    .background(.gray.opacity(0.8))
                    .foregroundStyle(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            case .running(let result):
                VStack(alignment: .trailing, spacing: 2) {
                    Text(String(format: "%.3f", result.score))
                        .font(.caption2.monospaced())
                        .foregroundStyle(.white.opacity(0.7))
                    Text(result.label == .anomaly ? "ANOMALY" : "GOOD")
                        .font(.caption.bold())
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(result.label == .anomaly ? Color.red : Color.green)
                        .foregroundStyle(.white)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                }
            case .error(let reason):
                Text(reason)
                    .font(.caption)
                    .foregroundStyle(.red)
            }
        }
    }
}
