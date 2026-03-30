import SwiftUI
import UIKit

/// Publishes the visual rotation angle that UI elements should apply to appear
/// upright when the device is held in a non-portrait orientation.
/// The app is locked to portrait at the system level; individual elements use
/// this angle with .rotationEffect() to simulate device rotation.
@MainActor
final class DeviceRotation: ObservableObject {
    static let shared = DeviceRotation()

    @Published private(set) var angle: Angle = .degrees(0)

    private init() {
        UIDevice.current.beginGeneratingDeviceOrientationNotifications()
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(orientationChanged),
            name: UIDevice.orientationDidChangeNotification,
            object: nil
        )
    }

    @objc private func orientationChanged() {
        let newAngle: Angle
        switch UIDevice.current.orientation {
        case .landscapeLeft:      newAngle = .degrees(90)
        case .landscapeRight:     newAngle = .degrees(-90)
        case .portraitUpsideDown: newAngle = .degrees(180)
        default:                  newAngle = .degrees(0)
        }
        withAnimation(.easeInOut(duration: 0.3)) {
            angle = newAngle
        }
    }
}
