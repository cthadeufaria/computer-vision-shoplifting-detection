# iOS STG-NF Shoplifting Detection App — TDD Implementation Plan

## Context

Port the STG-NF pose anomaly detection model into a native iOS proof-of-concept app. The app captures camera frames, detects poses via Apple's Vision framework, runs STG-NF normalizing-flow inference (via CoreML), and displays real-time GOOD/ANOMALY labels with skeleton overlay. Test-driven development: tests are written first, then implementation follows.

---

## Repository Structure

New `ios/` subtree within the existing repo:

```
ios/
├── project.yml                          # xcodegen spec → generates ShopliftDetect.xcodeproj
├── ShopliftDetect/
│   ├── App/
│   │   ├── ShopliftDetectApp.swift      # @main, routes Onboarding ↔ Home
│   │   └── AppEnvironment.swift
│   ├── Onboarding/
│   │   ├── OnboardingView.swift         # 3-page TabView: Welcome / How It Works / Camera Permission
│   │   ├── OnboardingPageView.swift
│   │   └── OnboardingViewModel.swift
│   ├── Home/
│   │   ├── HomeView.swift               # "Start Detection" toggle button
│   │   └── HomeViewModel.swift
│   ├── Detection/
│   │   ├── DetectionView.swift          # ZStack: camera + skeleton + score cards
│   │   ├── DetectionViewModel.swift     # full pipeline orchestration
│   │   ├── SkeletonOverlayView.swift    # Canvas-based bone drawing
│   │   └── ScoreCardView.swift          # score + GOOD/ANOMALY badge
│   ├── Core/
│   │   ├── Camera/
│   │   │   ├── CameraSession.swift      # AVCaptureSession → CVPixelBuffer publisher
│   │   │   └── CameraPreviewLayer.swift # UIViewRepresentable
│   │   ├── Pose/
│   │   │   ├── PoseEstimator.swift      # VNDetectHumanBodyPoseRequest
│   │   │   ├── KeypointConverter.swift  # COCO17→COCO18 + reorder + neck logic
│   │   │   └── PoseNormalizer.swift     # normalize_pose Swift port → MLMultiArray
│   │   ├── Model/
│   │   │   ├── STGNFModel.swift         # CoreML wrapper
│   │   │   ├── FrameBuffer.swift        # 24-frame rolling window (Swift actor)
│   │   │   └── AnomalyScorer.swift      # NLL → anomaly_score = -NLL → label
│   │   └── Domain/
│   │       ├── Keypoint.swift
│   │       ├── PoseSkeleton.swift
│   │       ├── AnomalyResult.swift
│   │       └── DetectionState.swift
│   └── Resources/
│       ├── Assets.xcassets
│       ├── STGNFModel.mlpackage/        # CoreML model (bundled)
│       └── Info.plist
├── ShopliftDetectTests/
│   ├── Pose/
│   │   ├── KeypointConverterTests.swift
│   │   └── PoseNormalizerTests.swift
│   ├── Model/
│   │   ├── FrameBufferTests.swift
│   │   ├── AnomalyScorerTests.swift
│   │   └── STGNFModelIntegrationTests.swift
│   └── Fixtures/
│       ├── coco17_sample.json            # raw 17-kp input + expected 18-kp output
│       └── normal_pose_window.json       # 24-frame input + expected normalized output
└── ShopliftDetectUITests/
    ├── OnboardingUITests.swift
    └── DetectionToggleUITests.swift

scripts/
└── convert_stgnf_to_coreml.py           # NEW: .pt → .mlpackage via ONNX intermediate

artifacts/stg_nf/coreml/
└── STGNFModel.mlpackage/                # output of conversion script
```

---

## Step 0: Xcode Project Bootstrap

```bash
cd ios
xcodegen generate          # reads project.yml → creates ShopliftDetect.xcodeproj
open ShopliftDetect.xcodeproj
```

**`project.yml` targets:**
- `ShopliftDetect` — iOS 17+, SwiftUI, Swift 6
- `ShopliftDetectTests` — XCTest unit tests, links ShopliftDetect target
- `ShopliftDetectUITests` — XCUITest UI tests

`Info.plist` must include `NSCameraUsageDescription`.

---

## Step 1: CoreML Conversion (before any Swift)

**Script:** `scripts/convert_stgnf_to_coreml.py`

Key steps inside the script:
1. Load `stg_nf_official/checkpoints/ShanghaiTech_85_9.tar` (PyTorch checkpoint)
2. Reconstruct `STG_NF` with hardcoded ShanghaiTech args: `pose_shape=(2,24,18)`, `K=8`, `L=1`, `R=3.0`, `flow_coupling='affine'`, `strategy='uniform'`
3. Call `model.load_state_dict(state_dict, strict=False)` then `model.set_actnorm_init()` — critical; skipping this corrupts ActNorm
4. Wrap in `STGNFWrapper(nn.Module)` that strips `label`/`score` params and returns `nll` as a 1D tensor
5. `torch.jit.trace(wrapper, example_input=torch.zeros(1,2,24,18))` → then `coremltools.convert(..., minimum_deployment_target=iOS17, compute_precision=FLOAT32)`
6. Numeric verification: assert Python vs CoreML NLL diff < 0.01 on random input
7. Save to `artifacts/stg_nf/coreml/STGNFModel.mlpackage`

Then: `cp -r artifacts/stg_nf/coreml/STGNFModel.mlpackage ios/ShopliftDetect/Resources/`

**CoreML I/O spec:**
- Input: `pose_window` — shape `[1, 2, 24, 18]` float32
- Output: `nll_score` — shape `[1]` float32 (Swift: `anomaly_score = -nll_score[0]`)

**Known risk:** `InvertibleConv1x1` + `torch.einsum` in ST-GCN may fail `jit.trace`. Fallback: export to ONNX first (`torch.onnx.export`) then `ct.convert(onnx_model)`.

---

## Step 2: Generate Fixtures (Python, before writing tests)

```bash
python3 - <<'EOF'
import sys, json, numpy as np
sys.path.insert(0, 'stg_nf_official')
from dataset import keypoints17_to_coco18
from utils.data_utils import normalize_pose

# Fixture 1: keypoint conversion
coco17 = np.zeros((17, 3), dtype=np.float32)
coco17[0] = [100, 200, 0.9]   # nose
coco17[5] = [150, 300, 0.85]  # left_shoulder
coco17[6] = [250, 300, 0.88]  # right_shoulder
expected_neck = 0.5 * (coco17[5] + coco17[6])  # [200, 300, 0.865]
coco18 = keypoints17_to_coco18(coco17[None, None])[0, 0]  # [18, 3]
json.dump({'input': coco17.tolist(), 'output': coco18.tolist(),
           'expected_neck_xy': expected_neck[:2].tolist()},
          open('ios/ShopliftDetectTests/Fixtures/coco17_sample.json', 'w'))

# Fixture 2: normalization (seeded for reproducibility)
rng = np.random.RandomState(42)
window = rng.randn(1, 24, 18, 3).astype(np.float32)
window[..., 2] = np.clip(np.abs(window[..., 2]), 0.3, 1.0)
vid_res = [640, 480]
normalized = normalize_pose(window.copy(), vid_res=vid_res)
json.dump({'input': window.tolist(), 'vid_res': vid_res,
           'expected_output': normalized.tolist()},
          open('ios/ShopliftDetectTests/Fixtures/normal_pose_window.json', 'w'))
EOF
```

---

## Step 3: TDD — Write All Tests Before Implementation

### 3a. `KeypointConverterTests` (10 tests)

```swift
func testNeckIsSyntheticAverageOfShoulders()         // COCO17 idx5+idx6 / 2
func testOutputHas18Keypoints()
func testOpenPoseReorderingMatchesPythonOppOrder()   // opp_order = [0,17,6,8,10,5,7,9,12,14,16,11,13,15,2,1,4,3]
func testNoseRemainsAtIndex0()
func testRightShoulderLandsAtOpenPoseIndex2()
func testLeftShoulderLandsAtOpenPoseIndex5()
func testUsesVisionNeckWhenHighConfidence()           // if neck conf >= 0.3, use it directly
func testFallsBackToSyntheticNeckWhenLowConfidence()
func testConfidenceValuesPreservedThroughConversion()
func testZeroInputProducesZeroNeck()
```

### 3b. `PoseNormalizerTests` (8 tests)

```swift
func testResolutionDivisionByWidthAndHeight()        // x/W, y/H before anything
func testMeanSubtractionOverAllFramesAndJoints()     // mean of [24×18] xy → ~0
func testYAxisStdAppliedToBothXandY()                // std computed on y only
func testConfidencePassesThroughUnmodified()
func testMatchesPythonOutputForSeededFixture()       // 1e-5 tolerance, uses Fixtures/normal_pose_window.json
func testZeroVarianceDoesNotProduceNaN()
func testOutputShapeIs_1_2_24_18()                   // drops conf → [1,2,24,18]
func testUsesPixelCoordinatesNotNormalized()
```

### 3c. `FrameBufferTests` (8 tests)

```swift
func testEmptyBufferIsNotReady()
func testPartialBufferIsNotReadyBelow24()
func testAt24FramesBufferIsReady()
func testFrame25EvintsOldestFrame()                  // FIFO ring buffer
func testFIFOOrderingPreserved()
func testResetClearsFrames()
func testExportedTensorShapeIs_2_24_18()
func testConcurrentAccessIsThreadSafe()              // Swift actor isolation
```

### 3d. `AnomalyScorerTests` (8 tests)

```swift
func testScoreBelowThresholdIsAnomaly()
func testScoreAboveThresholdIsNormal()
func testScoreAtThresholdIsAnomaly()                 // boundary: score == threshold → ANOMALY
func testDefaultThresholdIsNegative1Point2()
func testThresholdIsSettable()
func testResultContainsScoreAndTimestamp()
func testWarmupFlagTrueWhenBufferNotFull()
func testWarmupFlagFalseAfterFirstFullWindow()
```

### 3e. `STGNFModelIntegrationTests` (5 tests, skip-guarded until .mlpackage present)

```swift
func testModelLoadsFromBundle()
func testModelOutputIsFiniteOnZeroInput()
func testNormalPoseFixtureProducesHighScore()
func testSingleWindowInferenceUnder50ms()
func testCoreMLMatchesPythonNLLWithin1e3()           // compare vs Python on same random seed
```

### 3f. `OnboardingUITests` (5 tests)

```swift
func testFirstScreenIsWelcome()
func testNextButtonAdvancesToPageTwo()
func testThirdScreenHasPermissionCTA()
func testCanCompleteOnboardingFlow()
func testOnboardingSkippedOnSecondLaunch()           // UserDefaults gate
```

### 3g. `DetectionToggleUITests` (4 tests)

```swift
func testHomeShowsStartDetectionButton()
func testStartDetectionPresentsDetectionView()
func testWarmupIndicatorVisibleOnLaunch()
func testDismissReturnsToHome()
```

---

## Step 4: Domain Types

```swift
// Keypoint.swift — pixel coords, confidence
struct Keypoint: Sendable { let x, y, confidence: Float }

// PoseSkeleton.swift — 18 keypoints in OpenPose COCO18 order
struct PoseSkeleton: Sendable {
    let keypoints: [Keypoint]; let frameIndex: Int; let timestamp: CMTime
}

// AnomalyResult.swift
enum AnomalyLabel: Sendable { case normal, anomaly, warmup }
struct AnomalyResult: Sendable { let score: Float; let label: AnomalyLabel; let timestamp: Date }

// DetectionState.swift
enum DetectionState: Sendable {
    case idle
    case warmingUp(framesCollected: Int, framesNeeded: Int)
    case running(latestResult: AnomalyResult)
    case error(reason: String)
}
```

---

## Step 5: Core Implementations

### Coordinate Space (critical)

Vision gives normalized `(0,0)=bottom-left`. Convert before `KeypointConverter`:
```swift
let pixelX = Float(point.x) * Float(previewSize.width)
let pixelY = Float(1.0 - point.y) * Float(previewSize.height)
```

### `KeypointConverter`

`opp_order = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]`

Neck strategy: if `VNHumanBodyPoseObservation` `.neck` confidence >= 0.3, use it directly; otherwise compute `0.5 * (leftShoulder + rightShoulder)`.

### `PoseNormalizer` → `MLMultiArray`

Exactly matches `data_utils.py:normalize_pose`:
1. Divide all x by width, all y by height, conf unchanged
2. Compute spatial mean over all 24×18 joint xy pairs
3. Subtract mean from xy
4. Compute std on y-column only (all 24×18 y values)
5. Divide both x and y by that scalar std
6. Transpose to `[1, 2, 24, 18]`, drop conf channel → `MLMultiArray`

### `FrameBuffer` (actor)

Ring buffer of `[PoseSkeleton]`, capacity 24. `currentWindow() -> [PoseSkeleton]?` returns `nil` until full.

### `STGNFModel`

```swift
final class STGNFModel: STGNFModelProtocol {
    private let model: MLModel
    init() throws { model = try STGNFModel_generated(configuration: .init()).model }
    func runInference(on input: MLMultiArray) throws -> Float {
        let result = try model.prediction(from: STGNFModelInput(pose_window: input))
        let nll = result.featureValue(for: "nll_score")!.multiArrayValue![0].floatValue
        return -nll  // anomaly_score = -NLL
    }
}
```

### `AnomalyScorer`

```swift
struct AnomalyScorer {
    var threshold: Float = -1.2
    func classify(score: Float, isWarmup: Bool) -> AnomalyResult {
        let label: AnomalyLabel = isWarmup ? .warmup : (score <= threshold ? .anomaly : .normal)
        return AnomalyResult(score: score, label: label, timestamp: Date())
    }
}
```

---

## Step 6: Detection Pipeline (DetectionViewModel)

```
CameraSession.pixelBufferPublisher (30fps Combine stream)
  → PoseEstimator.detectPose(pixelBuffer)   // VNDetectHumanBodyPoseRequest
  → for each person: KeypointConverter.convert(observation, previewSize, frameIndex)
  → per-person FrameBuffer.append(skeleton)  // keyed by bounding-box IoU tracking
  → if buffer.isReady: PoseNormalizer.normalize(buffer.currentWindow()!)
  → STGNFModel.runInference(mlArray)
  → AnomalyScorer.classify(score, isWarmup: false)
  → @MainActor publish detectionState + skeletons
```

Multi-person: maintain `[String: FrameBuffer]` keyed by IoU-matched bounding-box track IDs (mirrors Python `SimpleIoUTracker` with `iou_threshold=0.3`, `max_missing=6`).

---

## Step 7: UI Assembly

- **OnboardingView**: 3-page `TabView` (Welcome, How It Works, Camera Permission) — UserDefaults gate; skipped on second launch
- **HomeView**: single "Start Detection" button → presents `DetectionView` full-screen
- **DetectionView** ZStack layers (bottom to top):
  1. `CameraPreviewLayer` (full screen)
  2. `SkeletonOverlayView` (Canvas, draws 17 bones connecting OpenPose18 joints)
  3. `ScoreCardView` per tracked person (top-right area): score + **GOOD** (green) / **ANOMALY** (red) / **Warming up X/24** (gray)
  4. `WarmupIndicatorView` (centered, shown only during warmup)
  5. Dismiss button (top-left)

---

## Implementation Order

| Step | Deliverable | Gate |
|------|-------------|------|
| 1 | `project.yml` + `xcodegen generate` | Compiles clean |
| 2 | `convert_stgnf_to_coreml.py` → `.mlpackage` | Numeric verification passes |
| 3 | Fixture JSON files from Python | Files written, values spot-checked |
| 4 | All test files written (failing) | Tests compile, all fail |
| 5 | Domain types (`Keypoint`, `PoseSkeleton`, etc.) | — |
| 6 | `KeypointConverter` | `KeypointConverterTests` all pass |
| 7 | `PoseNormalizer` | `PoseNormalizerTests` all pass incl. fixture test |
| 8 | `FrameBuffer` | `FrameBufferTests` all pass |
| 9 | `AnomalyScorer` | `AnomalyScorerTests` all pass |
| 10 | `STGNFModel` | Integration tests pass (especially `testCoreMLMatchesPythonNLL`) |
| 11 | `CameraSession` + `PoseEstimator` | Manual device test: skeleton visible |
| 12 | `DetectionViewModel` | Manual device test: scores display live |
| 13 | Detection UI: skeleton overlay + score cards | Manual device test |
| 14 | Onboarding + Home | `OnboardingUITests` + `DetectionToggleUITests` pass |
| 15 | Full regression | All unit + integration + UI tests pass |

---

## Critical Files to Reference During Implementation

| File | What to copy from it |
|------|---------------------|
| `stg_nf_official/dataset.py:keypoints17_to_coco18` | Exact `opp_order` array + neck averaging logic |
| `stg_nf_official/utils/data_utils.py:normalize_pose` | Mean/std normalization formula (use the live branch, not the dead commented-out one above it) |
| `stg_nf_official/models/STG_NF/model_pose.py:STG_NF.normal_flow` | Confirms output is already NLL; `anomaly_score = -1 * nll` |
| `stg_nf_official/args.py` | Default hyperparameters for conversion script (`seg_len=24`, `K=8`, `L=1`, `R=3.0`) |
| `pipeline/video_inference_pipeline.py` | IoU tracking logic (`iou_threshold=0.3`, `max_missing=6`) for multi-person DetectionViewModel |

---

## Verification

1. **Unit tests:** `Cmd+U` in Xcode — all 43 unit + integration tests pass
2. **UI tests:** Run on simulator (onboarding/toggle) + device (camera tests)
3. **Numeric fidelity:** `STGNFModelIntegrationTests.testCoreMLMatchesPythonNLLWithin1e3` passes
4. **Device smoke test:** Open app on real iPhone → complete onboarding → tap Start Detection → walk in front of camera → skeleton overlay appears → score card shows "Warming up" for ~0.8 seconds → then GOOD/ANOMALY label appears in real time
5. **Domain mismatch note:** With the ShanghaiTech model, normal retail footage may trigger ANOMALY — this is expected and documented in the UI with a "Threshold calibration needed" note

---

## Model Import Strategy: CoreML vs ONNX Runtime

**Recommended: CoreML (with ONNX as intermediate)**

| Option | Pros | Cons |
|--------|------|------|
| **CoreML** (via `coremltools` from ONNX) | Uses Apple Neural Engine (ANE), best battery + perf, native Xcode integration, no extra dependency | Extra conversion step, coremltools version sensitivity |
| **ONNX Runtime for iOS** (via SPM `onnxruntime-objc`) | Skips CoreML conversion, exact numeric parity with Python, portable to Android | No ANE access, CPU/GPU only, larger binary (+~30MB) |

**Decision:** Use CoreML with ONNX as the **intermediate format** (`torch.onnx.export` → `ct.convert(onnx_path)`). This gives the best of both: simpler conversion than direct TorchScript tracing (avoids dynamic control-flow issues), and native ANE acceleration at runtime. If conversion still fails, fall back to shipping the ONNX model directly with `onnxruntime-objc`.

---

## Language Strategy: Swift vs Rust

### POC Phase — Swift only

**The entire app is written in Swift.** No Rust in this phase.

The layer responsibilities are:
1. **Vision framework** — pose detection (`VNDetectHumanBodyPoseRequest`)
2. **Swift** — keypoint conversion, normalization, frame buffering, scoring logic, all UI
3. **CoreML** — neural net inference (handled by Apple's framework internally; Swift just calls it)

The preprocessing math (~50 lines of Float arithmetic) does not justify a Rust dependency at this stage. Swift is faster to build, easier to test with XCTest, and sufficient for the POC workload.

### Post-POC — Where Rust fits in

As the algorithm grows, a Rust compute kernel becomes a legitimate upgrade for the layer between Vision and CoreML:

```
Vision (Swift)          → raw 17 keypoints
Rust kernel (.a lib)    → COCO17→COCO18, normalization, IoU tracker,
                           Kalman filter, rolling window, score smoothing
                        → [1, 2, 24, 18] f32 tensor
CoreML (Swift call)     → NLL score
Rust scoring logic      → threshold, debouncing, alert history
Swift / SwiftUI         → render UI
```

**Why Rust for that middle layer:**
- No garbage collector — no frame-drop spikes during tracking
- `cargo test` runs entirely on macOS, no simulator needed — fast TDD loop for math-heavy code
- Compiles to a static `.a` via `cargo build --target aarch64-apple-ios` — zero runtime overhead on device
- Portable: same crate compiles to `.so` for Android, enabling a cross-platform SDK later
- Bridge: expose a C ABI (`#[no_mangle] extern "C" fn`) and import via Swift bridging header, or use `UniFFI` to auto-generate Swift bindings

**Trigger for this upgrade:** when the preprocessing pipeline expands beyond the POC scope — specifically when adding multi-person IoU tracking, Kalman filtering per track, or temporal score windowing. At that point, migrate `KeypointConverter`, `PoseNormalizer`, `FrameBuffer`, and `AnomalyScorer` to a Rust crate (`ios/ShopliftDetectCore/`) while keeping all Swift UI and CoreML call sites unchanged.

**For the POC:** Swift + CoreML, full stop. The architecture is designed so the Swift types (`Keypoint`, `PoseSkeleton`, `AnomalyResult`) become the FFI boundary when the Rust migration happens — no UI or CoreML changes required.

---

## Known Issues / Notes

- **Domain mismatch:** ShanghaiTech_85_9 was trained on campus anomalies. Normal retail footage may score as anomalous. The threshold (`-1.2`) should be UserDefaults-backed and exposed in a Settings sheet for calibration.
- **Simulator:** Camera unavailable on simulator — Camera + Pose tests must run on device. Integration tests and unit tests run fine on simulator.
- **CoreML conversion risk:** If `torch.jit.trace` fails due to dynamic control flow in `FlowStep`, use ONNX export as intermediate: `torch.onnx.export(wrapper, ...)` → `ct.convert(onnx_model_path, ...)`.
- **Apple Vision neck joint:** `VNHumanBodyPoseObservationJointNameNeck` exists in the API — use it when confidence >= 0.3 to avoid the synthetic averaging error.
