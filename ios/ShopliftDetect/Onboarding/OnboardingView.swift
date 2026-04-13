import SwiftUI

struct OnboardingView: View {
    @EnvironmentObject private var appEnvironment: AppEnvironment
    @StateObject private var viewModel: OnboardingViewModel

    init(viewModel: @autoclosure @escaping () -> OnboardingViewModel) {
        _viewModel = StateObject(wrappedValue: viewModel())
    }

    var body: some View {
        TabView(selection: $viewModel.currentPage) {
            OnboardingPageView(
                title: "Welcome",
                description: "Privacy-preserving shoplifting detection using pose estimation — no video stored.",
                systemImage: "figure.walk"
            )
            .tag(0)

            OnboardingPageView(
                title: "How It Works",
                description: "Your camera detects body poses in real time. The AI analyses movement patterns to flag anomalies — without recording any video.",
                systemImage: "cpu"
            )
            .tag(1)

            RoleSelectionView(
                selectedRole: viewModel.selectedRole,
                cameraSubtitle: viewModel.roleSubtitle(for: .camera),
                supervisorSubtitle: viewModel.roleSubtitle(for: .supervisor),
                supportsSupervisorRole: viewModel.supportsRole(.supervisor),
                supervisorAvailabilityNote: viewModel.supervisorAvailabilityNote,
                onSelect: viewModel.selectRole
            )
            .tag(2)

            VStack(spacing: 24) {
                OnboardingPageView(
                    title: viewModel.selectedRoleTitle(),
                    description: viewModel.permissionSummaryText(),
                    systemImage: viewModel.selectedRole == .camera ? "qrcode" : "qrcode.viewfinder"
                )

                AppearancePickerView(selectedAppearance: viewModel.selectedAppearance) { appearance in
                    viewModel.selectAppearance(appearance)
                }
                .accessibilityIdentifier("onboardingAppearancePicker")

                if viewModel.selectedRole == .camera {
                    QRCodeDisplayView(
                        payload: viewModel.qrPayload,
                        connectionStateText: viewModel.connectionStateText()
                    )
                } else if viewModel.selectedRole == .supervisor {
                    QRScannerView(
                        payloadText: $viewModel.scannedPayload,
                        connectionStateText: viewModel.connectionStateText(),
                        onScan: viewModel.scanQRCode
                    )
                }

                Button(viewModel.permissionButtonTitle()) {
                    Task {
                        await viewModel.completeAfterPermissions()
                        appEnvironment.refreshOnboardingState()
                    }
                }
                .buttonStyle(.borderedProminent)
                .accessibilityIdentifier("grantCameraAccessButton")
            }
            .tag(3)
        }
        .tabViewStyle(.page(indexDisplayMode: .always))
        .animation(.easeInOut, value: viewModel.currentPage)
        .onAppear {
            viewModel.updatePairingScreenVisibility(isVisible: viewModel.currentPage == viewModel.totalPages - 1)
        }
        .onChange(of: viewModel.currentPage) { newValue in
            viewModel.updatePairingScreenVisibility(isVisible: newValue == viewModel.totalPages - 1)
        }
        .overlay(alignment: .bottomTrailing) {
            if viewModel.currentPage < viewModel.totalPages - 1 {
                Button("Next") {
                    withAnimation { viewModel.nextPage() }
                }
                .buttonStyle(.bordered)
                .disabled(!viewModel.canAdvance)
                .padding()
                .accessibilityIdentifier("nextButton")
            }
        }
        .alert("Setup Incomplete", isPresented: Binding(
            get: { viewModel.errorMessage != nil },
            set: { if !$0 { viewModel.errorMessage = nil } }
        )) {
            Button("OK") {}
        } message: {
            Text(viewModel.errorMessage ?? "")
        }
        .screenAppearanceIdentifier("onboardingScreen")
    }
}
