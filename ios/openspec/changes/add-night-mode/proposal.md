## Why

The app currently relies on default system styling, which leaves several key screens visually inconsistent and hard to use in low-light retail environments. Adding a first-class night mode now provides a predictable dark appearance across onboarding, camera, and supervisor flows while establishing a reusable theme preference for future UI work.

## What Changes

- Add a user-selectable night mode preference that can be persisted across launches.
- Apply the selected appearance consistently to onboarding, home, detection, pose preview, networking, and supervisor screens.
- Introduce a shared theme abstraction in app composition so views consume appearance state without embedding business logic.
- Add unit and UI test coverage for theme persistence, app startup application, and the primary user flows rendered in night mode.

## Capabilities

### New Capabilities
- `theme-preference`: Persist and apply an explicit light or dark appearance across the app, with UI surfaces updating from a shared app-level theme state.

### Modified Capabilities

## Impact

- Affects `AppEnvironment`, `ShopliftDetectApp`, `SettingsService`, persistence-backed settings models, and the main SwiftUI screen hierarchy.
- Requires updates to view models and tests that currently assume only anomaly-threshold settings are stored.
- Adds UI verification for dark appearance on onboarding, detection, pose preview, and supervisor experiences.
