## ADDED Requirements

### Requirement: User can select app appearance
The system SHALL allow the user to switch the app between light mode and dark mode from an existing user-facing flow without requiring the app to restart.

#### Scenario: Select dark mode during onboarding
- **WHEN** a user chooses dark mode from the onboarding flow
- **THEN** the app SHALL update the active appearance to dark mode immediately
- **THEN** the selected appearance SHALL remain available after onboarding completes

#### Scenario: Change appearance after onboarding
- **WHEN** a user changes the appearance from an existing post-onboarding screen
- **THEN** the app SHALL apply the new appearance to the current screen and presented flows
- **THEN** the system SHALL preserve the new preference for future launches

### Requirement: Appearance preference persists across launches
The system SHALL persist the selected app appearance through the settings and persistence services so the same appearance is restored on the next launch.

#### Scenario: Restore persisted dark mode
- **WHEN** the stored appearance preference is dark mode at app launch
- **THEN** the app SHALL start in dark mode before the primary content is shown

#### Scenario: Default appearance when no preference exists
- **WHEN** no appearance preference has been stored yet
- **THEN** the app SHALL start in light mode by default

### Requirement: Primary screens support dark appearance
The system SHALL render the onboarding, home, detection, pose preview, networking, and supervisor flows with legible colors and controls when dark mode is active.

#### Scenario: Detection flow in dark mode
- **WHEN** dark mode is active and the user opens detection or pose preview
- **THEN** overlays, controls, and supporting panels SHALL remain legible against dark surfaces and camera content

#### Scenario: Supervisor flow in dark mode
- **WHEN** dark mode is active and the user views the supervisor grid or feed detail
- **THEN** banners, empty states, tiles, and navigation chrome SHALL remain readable and visually consistent with dark mode
