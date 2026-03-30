import SwiftUI

struct OnboardingPageView: View {
    let title: String
    let description: String
    let systemImage: String
    var body: some View {
        VStack(spacing: 24) {
            Image(systemName: systemImage)
                .font(.system(size: 80))
                .foregroundStyle(.blue)
            Text(title)
                .font(.title.bold())
                .multilineTextAlignment(.center)
            Text(description)
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
        }
        .padding()
    }
}
