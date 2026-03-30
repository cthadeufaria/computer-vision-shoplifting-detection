import SwiftUI

struct DetectionScoreCardOverlay: View {
    let state: DetectionState
    let rotation: Angle

    var body: some View {
        VStack {
            HStack {
                Spacer()
                ScoreCardView(state: state)
                    .rotationEffect(rotation)
                    .padding()
            }
            Spacer()
        }
    }
}

struct WarmupIndicatorView: View {
    let state: DetectionState
    let rotation: Angle

    var body: some View {
        if case .warmingUp(let collected, let needed) = state {
            Text("Collecting frames \(collected)/\(needed)")
                .font(.headline)
                .padding()
                .background(.ultraThinMaterial)
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .accessibilityIdentifier("warmupIndicator")
                .rotationEffect(rotation)
        }
    }
}

struct DetectionDismissButton: View {
    let rotation: Angle
    let action: () -> Void

    var body: some View {
        VStack {
            HStack {
                Button(action: action) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title)
                        .foregroundStyle(.white)
                }
                .accessibilityIdentifier("xmark.circle.fill")
                .rotationEffect(rotation)
                .padding()
                Spacer()
            }
            Spacer()
        }
    }
}
