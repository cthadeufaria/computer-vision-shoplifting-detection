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

struct DetectionStatusOverlay: View {
    let threshold: Float
    let isStreaming: Bool
    let rotation: Angle
    let decreaseThreshold: () -> Void
    let increaseThreshold: () -> Void

    var body: some View {
        VStack {
            Spacer()
            HStack {
                thresholdControls
                Spacer()
                streamingBadge
            }
            .padding()
        }
    }

    private var thresholdControls: some View {
        HStack(spacing: 12) {
            Button(action: decreaseThreshold) {
                Image(systemName: "minus")
                    .font(.headline)
                    .frame(width: 32, height: 32)
            }
            .buttonStyle(.borderedProminent)
            .accessibilityIdentifier("decreaseThresholdButton")

            Text(String(format: "Threshold %.1f", threshold))
                .font(.subheadline.monospacedDigit())
                .accessibilityIdentifier("thresholdValueLabel")

            Button(action: increaseThreshold) {
                Image(systemName: "plus")
                    .font(.headline)
                    .frame(width: 32, height: 32)
            }
            .buttonStyle(.borderedProminent)
            .accessibilityIdentifier("increaseThresholdButton")
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 14))
        .rotationEffect(rotation)
    }

    private var streamingBadge: some View {
        Label(isStreaming ? "Streaming live" : "Streaming paused", systemImage: isStreaming ? "dot.radiowaves.left.and.right" : "pause.circle")
            .font(.caption.weight(.semibold))
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(.ultraThinMaterial)
            .clipShape(Capsule())
            .accessibilityIdentifier("streamingStatusLabel")
            .rotationEffect(rotation)
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
