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

## 4. Run the unit suite before and during implementation

```bash
xcodebuild test -project ShopliftDetect.xcodeproj -scheme ShopliftDetect \
  -destination 'platform=iOS Simulator,name=iPhone 17' \
  -only-testing:ShopliftDetectTests
```

## 5. Run the full suite before feature completion

```bash
xcodebuild test -project ShopliftDetect.xcodeproj -scheme ShopliftDetect \
  -destination 'platform=iOS Simulator,name=iPhone 17'
```

## 6. Validate the end-to-end workflow on devices

1. Launch device A and select `Smart Camera`.
2. Confirm the pairing screen displays a QR code with a fresh token.
3. Launch device B and select `Supervisory View`.
4. Scan the QR code and confirm authenticated connection.
5. Start detection on the camera device and confirm the supervisor receives frames and anomaly overlays.
6. Disconnect the camera or background it and confirm the supervisor freezes the last frame with a stale/disconnected overlay within 5 seconds.
