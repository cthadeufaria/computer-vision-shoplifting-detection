import Foundation

@MainActor
final class AppEnvironment: ObservableObject {
    @Published private(set) var onboardingComplete: Bool

    let persistenceService: PersistenceServiceProtocol
    let permissionService: PermissionServiceProtocol
    let settingsService: SettingsServiceProtocol
    let pairingService: PairingServiceProtocol
    let streamingService: StreamingServiceProtocol

    init(
        persistenceService: PersistenceServiceProtocol = UserDefaultsPersistenceService(),
        permissionService: PermissionServiceProtocol = AVPermissionService(),
        settingsService: SettingsServiceProtocol? = nil,
        pairingService: PairingServiceProtocol = NoopPairingService(),
        streamingService: StreamingServiceProtocol = NoopStreamingService()
    ) {
        self.persistenceService = persistenceService
        self.permissionService = permissionService
        self.settingsService = settingsService ?? UserDefaultsSettingsService(persistence: persistenceService)
        self.pairingService = pairingService
        self.streamingService = streamingService
        self.onboardingComplete = persistenceService.onboardingComplete
    }

    func applyLaunchArguments(_ arguments: [String]) {
        if arguments.contains("--reset-onboarding") {
            persistenceService.onboardingComplete = false
            persistenceService.selectedRole = nil
        } else if arguments.contains("--skip-onboarding") {
            persistenceService.onboardingComplete = true
        }

        refreshOnboardingState()
    }

    func refreshOnboardingState() {
        onboardingComplete = persistenceService.onboardingComplete
    }

    func makeHomeViewModel() -> HomeViewModel {
        HomeViewModel(
            persistence: persistenceService,
            settings: settingsService
        )
    }

    func makeOnboardingViewModel() -> OnboardingViewModel {
        OnboardingViewModel(
            persistence: persistenceService,
            permission: permissionService
        )
    }

    func makeDetectionViewModel() -> DetectionViewModel {
        DetectionViewModel(
            camera: CameraSession(),
            estimator: PoseEstimator(),
            converter: KeypointConverter(),
            scorer: AnomalyScorer(threshold: settingsService.anomalyThreshold),
            tracking: TrackingService(),
            settings: settingsService,
            streaming: streamingService
        )
    }

    func makePosePreviewViewModel() -> PosePreviewViewModel {
        PosePreviewViewModel(
            camera: CameraSession(),
            estimator: PoseEstimator(),
            converter: KeypointConverter()
        )
    }
}
