import Foundation

@MainActor
final class AppEnvironment: ObservableObject {
    @Published private(set) var onboardingComplete: Bool
    private(set) var launchArguments: [String]

    let persistenceService: PersistenceServiceProtocol
    let permissionService: PermissionServiceProtocol
    let settingsService: SettingsServiceProtocol
    let pairingService: PairingServiceProtocol
    let streamingService: StreamingServiceProtocol

    init(
        persistenceService: PersistenceServiceProtocol = UserDefaultsPersistenceService(),
        permissionService: PermissionServiceProtocol = AVPermissionService(),
        settingsService: SettingsServiceProtocol? = nil,
        pairingService: PairingServiceProtocol = PairingService(),
        streamingService: StreamingServiceProtocol = StreamingService(),
        launchArguments: [String] = []
    ) {
        self.persistenceService = persistenceService
        self.permissionService = permissionService
        self.settingsService = settingsService ?? UserDefaultsSettingsService(persistence: persistenceService)
        self.pairingService = pairingService
        self.streamingService = streamingService
        self.onboardingComplete = persistenceService.onboardingComplete
        self.launchArguments = launchArguments
    }

    func applyLaunchArguments(_ arguments: [String]) {
        launchArguments = arguments

        if arguments.contains("--reset-onboarding") {
            persistenceService.onboardingComplete = false
            persistenceService.selectedRole = nil
        } else if arguments.contains("--skip-onboarding") {
            persistenceService.onboardingComplete = true
            if !arguments.contains("--ui-test-supervisor-role") {
                persistenceService.selectedRole = .camera
            }
        }

        if arguments.contains("--ui-test-supervisor-role") {
            persistenceService.onboardingComplete = true
            persistenceService.selectedRole = .supervisor
        }

        configureUITestSupervisorState(arguments)

        refreshOnboardingState()
    }

    func refreshOnboardingState() {
        onboardingComplete = persistenceService.onboardingComplete
    }

    func makeHomeViewModel() -> HomeViewModel {
        HomeViewModel(
            persistence: persistenceService,
            settings: settingsService,
            pairing: pairingService
        )
    }

    func makeOnboardingViewModel() -> OnboardingViewModel {
        OnboardingViewModel(
            persistence: persistenceService,
            permission: permissionService,
            pairing: pairingService,
            prefilledSupervisorPayload: launchArgumentValue(prefix: "--ui-test-pairing-payload=")
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

    func makeSupervisorViewModel() -> SupervisorViewModel {
        SupervisorViewModel(
            pairing: pairingService,
            streaming: streamingService
        )
    }

    private func launchArgumentValue(prefix: String) -> String? {
        launchArguments.first(where: { $0.hasPrefix(prefix) })?.replacingOccurrences(of: prefix, with: "")
    }

    private func configureUITestSupervisorState(_ arguments: [String]) {
        guard let pairingService = pairingService as? PairingService else { return }

        if arguments.contains("--ui-test-supervisor-maxed") {
            pairingService.seedSupervisorSessions(deviceNames: [
                "Aisle 1 Camera",
                "Aisle 2 Camera",
                "Aisle 3 Camera",
                "Aisle 4 Camera"
            ])
        } else if arguments.contains("--ui-test-supervisor-feed") {
            pairingService.seedSupervisorSessions(deviceNames: ["Aisle 3 Camera"])
        }

        if arguments.contains("--ui-test-supervisor-feed"),
           let session = pairingService.sessions.first {
            streamingService.registerFeed(session)
            streamingService.noteConnectionEstablished(at: Date())
            streamingService.publishFrame(
                VideoFrame(timestamp: 1, jpegData: Data([0x01, 0x02]), width: 120, height: 90),
                for: session.sessionID
            )
            streamingService.publishDetections(
                [
                    DetectionResult(
                        trackID: 7,
                        score: -1.6,
                        label: .anomaly,
                        keypoints: [],
                        boundingBox: .zero,
                        timestamp: Date()
                    )
                ],
                for: session.sessionID
            )
        }
    }
}
