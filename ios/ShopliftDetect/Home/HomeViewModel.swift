import SwiftUI

@MainActor
final class HomeViewModel: ObservableObject {
    @Published var isDetectionActive = false
    @Published var isPosePreviewActive = false
}
