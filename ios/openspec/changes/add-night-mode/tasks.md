## 1. Settings And Persistence

- [x] 1.1 Add an app appearance domain model and extend `SettingsServiceProtocol` to read and write the persisted appearance preference.
- [x] 1.2 Update the persistence-backed settings implementation and test doubles to default to light mode and restore any saved appearance value.
- [x] 1.3 Add unit tests for settings and persistence behavior covering default appearance, saving dark mode, and restoring it on next launch.

## 2. App Composition And View Models

- [x] 2.1 Update `AppEnvironment` to publish the active appearance and expose an intent for changing it through injected services.
- [x] 2.2 Apply the active appearance at the `ShopliftDetectApp` root so onboarding, home, detection, pose preview, networking, and supervisor flows inherit the same color scheme.
- [x] 2.3 Extend the relevant view models to surface appearance state and handle user actions without allowing views to mutate services directly.

## 3. UI Surfaces

- [x] 3.1 Add a reusable appearance control to the onboarding completion flow and the existing post-onboarding landing experience.
- [x] 3.2 Audit the primary screens for hard-coded colors and update affected components so dark mode remains legible and visually consistent.
- [x] 3.3 Verify presented flows, including full-screen covers and supervisor detail screens, react immediately when the user switches appearance.

## 4. Verification

- [x] 4.1 Add or update unit tests for `AppEnvironment` and the affected view models to cover startup appearance application and appearance-change intents.
- [x] 4.2 Add UI tests that exercise selecting dark mode and navigating through onboarding, detection, pose preview, and supervisor flows with night mode active.
- [x] 4.3 Run the required `ShopliftDetectTests` suite and any impacted UI tests, then fix any regressions before implementation is considered complete.
