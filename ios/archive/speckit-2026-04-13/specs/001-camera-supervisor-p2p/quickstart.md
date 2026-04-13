# Quickstart: iOS Smart Camera to Supervisory Device P2P App

## Prerequisites

- Xcode 16 with iOS 15 simulators and physical iOS devices available
- `xcodegen` installed
- Python 3 environment capable of regenerating the STG-NF CoreML package if needed
- Two physical iOS devices on the same Wi-Fi network for end-to-end pairing tests

## 1. Align the project with the constitution

Update [`project.yml`](/Users/bernese/git/computer-vision-shoplifting-detection/ios/project.yml) and the generated Xcode project to iOS 15.0 before implementing new feature work.

Expected settings:

- `options.deploymentTarget.iOS: "15.0"`
- `settings.base.IPHONEOS_DEPLOYMENT_TARGET: "15.0"`
- No unguarded use of iOS 16+ APIs

## 2. Verify the model asset source

The planning baseline uses:

- Checkpoint: `/Users/bernese/git/computer-vision-shoplifting-detection/artifacts/stg_nf/multi_runs/Multi/Apr01_1416/Apr01_1419__checkpoint.pth.tar`
- Args: `/Users/bernese/git/computer-vision-shoplifting-detection/artifacts/stg_nf/multi_runs/Multi/Apr01_1416/args.json`
- Bundled CoreML package: [`STGNFModel.mlpackage`](/Users/bernese/git/computer-vision-shoplifting-detection/ios/ShopliftDetect/Resources/STGNFModel.mlpackage)

If the CoreML package needs to be regenerated, update the conversion script to target iOS 15 and validate numeric parity before replacing the bundled package.

## 3. Regenerate the Xcode project if configuration changes

```bash
cd /Users/bernese/git/computer-vision-shoplifting-detection/ios
xcodegen generate
```

## 4. Run the required automated suites for SC-008

Required suite inventory:

- Unit suite: `ShopliftDetectTests`
- Full UI + unit suite: scheme-wide `ShopliftDetect` test run, which includes `ShopliftDetectUITests`
- Required high-signal integration coverage inside those runs:
  - `STGNFModelIntegrationTests`
  - `PoseNormalizerTests`
  - `PairingServiceTests`
  - `StreamProtocolTests`
  - `EncryptedTransportTests`

Run the unit suite before and during implementation:

```bash
xcodebuild test -project ShopliftDetect.xcodeproj -scheme ShopliftDetect \
  -destination 'platform=iOS Simulator,name=iPhone 17' \
  -only-testing:ShopliftDetectTests
```

Run the full suite before feature completion:

```bash
xcodebuild test -project ShopliftDetect.xcodeproj -scheme ShopliftDetect \
  -destination 'platform=iOS Simulator,name=iPhone 17'
```

## 6. Validate the end-to-end workflow on devices

1. Launch device A and select `Smart Camera`.
2. Confirm the pairing screen displays a QR code with a fresh token.
3. Launch device B and select `Supervisory View`.
4. Start a timer when scanning begins, scan the QR code, and confirm authenticated connection within 30 seconds for `SC-004`.
5. From the camera home screen, start a timer, enter `Pose Preview`, confirm live preview visibility, dismiss it, start detection, and confirm the full flow completes within 30 seconds for `SC-010`.
6. From the camera home screen, enter `Pose Preview` at least 10 times in a row and dismiss it each time; confirm there is no crash, hang, or black-screen camera session.
7. Leave `Pose Preview` open for at least 60 seconds on the physical device and confirm skeleton/debug updates continue without a memory-pressure termination or frozen preview.
8. Start detection on the camera device and confirm the supervisor receives frames and anomaly overlays.
9. Disconnect the camera or background it and confirm the supervisor freezes the last frame with a stale/disconnected overlay within 5 seconds.

## 7. Record validation evidence

Capture the following in `feature-coverage-matrix.md` and any follow-up notes in this file:

- Automated suite results:
  - unit suite pass/fail and date
  - full suite pass/fail and date
  - failures or skipped tests, if any
- Measured workflow times:
  - onboarding to first detection start
  - QR scan start to connected state
  - Pose Preview open to dismiss to successful detection start
- Device stability evidence:
  - Pose Preview repeated present/dismiss result
  - Pose Preview 60-second stability result
  - disconnect/stale overlay result
  - fps, inference latency, and memory observations if measured

## 8. Latest execution notes

- 2026-04-11: The targeted unit regression command for `PosePreviewViewModelTests`, `DetectionViewModelTests`, `PairingServiceTests`, and `SupervisorViewModelTests` ended with an early simulator bootstrap failure (`xcodebuild` exit code `65`) before assertion results were finalized.
- 2026-04-11: The targeted UI regression command for `PosePreviewUITests` and `OnboardingUITests` restarted multiple times after unexpected app exits. Several onboarding cases executed successfully during reruns, but the selected-suite command did not complete with one clean final summary.
- 2026-04-11: Treat `T059` as open until the required suite inventory in section 4 completes cleanly on the simulator, then copy final pass/fail results into `feature-coverage-matrix.md`.
