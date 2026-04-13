## Context

The app already centralizes composition in `AppEnvironment`, persists user-facing settings through `PersistenceService` and `SettingsService`, and builds feature-specific view models from that root. Visual appearance is currently left to default SwiftUI behavior, so screens mix system materials, fixed colors, and uncoordinated backgrounds without a persistent app-level appearance choice.

Night mode is a cross-cutting UI change because it affects onboarding, home, detection, pose preview, networking surfaces, and supervisor views. The implementation also needs to stay aligned with the repo rules: business logic must remain outside views, persistence must be protocol-backed, and tests must cover state transitions and user-visible flows.

## Goals / Non-Goals

**Goals:**
- Introduce a persisted app appearance preference that explicitly supports light and dark modes.
- Apply the selected appearance from app composition so all screens render consistently without per-screen duplication.
- Provide a small, reusable UI control for switching appearance in the existing user flows.
- Add unit and UI coverage for persistence, startup behavior, and night-mode rendering paths.

**Non-Goals:**
- Redesign the app’s visual language beyond the minimum updates needed for coherent dark presentation.
- Add a full settings screen or broader preference-management system.
- Support additional appearance modes beyond explicit light and dark in this change.

## Decisions

Use `SettingsService` as the domain boundary for appearance preference.
Rationale: the service already owns persisted user-adjustable settings and is injected through `AppEnvironment`. Extending it with an appearance property preserves MVVM boundaries and keeps `UserDefaults` access behind protocols. An alternative was storing the preference directly in `AppEnvironment`, but that would couple composition logic to persistence details and reduce testability.

Add a small theme model, such as `AppAppearance`, in the domain layer and persist it independently from anomaly-threshold data.
Rationale: theme preference is not detection-specific, so it should not be forced into `DetectionSettings`. A dedicated value type or enum avoids conflating UI preferences with inference settings and keeps future settings growth manageable. The alternative of adding a `darkModeEnabled` boolean to `DetectionSettings` is simpler short-term but produces an inaccurate model boundary.

Publish the current appearance from `AppEnvironment` and apply it at the `WindowGroup` root with `.preferredColorScheme(...)`.
Rationale: a single source of truth at the app root lets the entire SwiftUI tree react immediately when the preference changes, including modal flows like detection, pose preview, and supervisor detail. The alternative of adding `.preferredColorScheme` to each screen would be repetitive and easy to miss.

Expose appearance-changing intents through view models, not direct bindings to services from views.
Rationale: views should render state and forward actions only. The onboarding and home/supervisor view models can surface the current preference and a mutation API while the environment remains responsible for cross-screen propagation. A direct service mutation from SwiftUI controls would violate the architecture rules.

Keep the user-facing control minimal and place it where it already fits the current flow.
Rationale: the app has no dedicated settings surface. The least disruptive implementation is to add an appearance picker/toggle to onboarding completion and role landing screens, which gives first-run access and post-onboarding adjustment without introducing new navigation.

## Risks / Trade-offs

- [Risk] Incomplete root-level application could leave modal screens in the wrong appearance. → Mitigation: apply the selected color scheme at the app root and verify full-screen covers in UI tests.
- [Risk] Hard-coded colors such as the supervisor banner may have poor contrast in dark mode. → Mitigation: audit existing fixed colors touched by the main flows and replace them with semantic styling where needed.
- [Risk] Expanding `SettingsServiceProtocol` will break mocks and tests. → Mitigation: update mock services in the same change and add focused unit tests for default and persisted appearance values.
- [Risk] Adding controls to multiple screens can drift into duplicate presentation logic. → Mitigation: use one small reusable appearance control view backed by view-model-provided state.
