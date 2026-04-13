import SwiftUI

extension AppAppearance {
    var preferredColorScheme: ColorScheme {
        switch self {
        case .light:
            return .light
        case .dark:
            return .dark
        }
    }
}
