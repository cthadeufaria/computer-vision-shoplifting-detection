# iOS App — Class Diagram

```mermaid
classDiagram
    %% ─── App Layer ───────────────────────────────────────────────────────────
    class ShopliftDetectApp {
        <<App>>
        -onboardingComplete : Bool
        +body : Scene
    }
    class AppEnvironment {
        <<MainActor, singleton>>
        +shared : AppEnvironment
    }

    %% ─── Views ───────────────────────────────────────────────────────────────
    class HomeView {
        <<View>>
    }
    class OnboardingView {
        <<View>>
    }
    class OnboardingPageView {
        <<View>>
        +title : String
        +description : String
        +imageName : String
    }
    class DetectionView {
        <<View>>
        -startError : String?
        -isPreviewUITest : Bool
    }
    class PosePreviewView {
        <<View>>
        -startError : String?
    }
    class SkeletonOverlayView {
        <<View>>
        +skeletons : [PoseSkeleton]
    }
    class ScoreCardView {
        <<View>>
        +state : DetectionState
    }
    class CameraPreviewLayer {
        <<UIViewRepresentable>>
        +previewLayer : AVCaptureVideoPreviewLayer
    }

    %% ─── ViewModels ──────────────────────────────────────────────────────────
    class HomeViewModel {
        <<MainActor, ObservableObject>>
        +isDetectionActive : Bool
        +isPosePreviewActive : Bool
    }
    class OnboardingViewModel {
        <<MainActor, ObservableObject>>
        +currentPage : Int
        +totalPages : Int
        +onboardingComplete : Bool
        +requestCameraPermission() async
        +complete()
    }
    class DetectionViewModel {
        <<MainActor, ObservableObject>>
        +detectionState : DetectionState
        +skeletons : [PoseSkeleton]
        +previewLayer : AVCaptureVideoPreviewLayer
        +start() throws
        +stop()
        +enablePreviewTestMode()
        -matchTrack(PoseSkeleton) String
        -computeIoU(CGRect, CGRect) CGFloat
        -processFrame(CVPixelBuffer) async
    }
    class PosePreviewViewModel {
        <<MainActor, ObservableObject>>
        +skeletons : [PoseSkeleton]
        +previewLayer : AVCaptureVideoPreviewLayer
        +start() throws
        +stop()
        -processFrame(CVPixelBuffer) async
    }

    %% ─── Services ────────────────────────────────────────────────────────────
    class CameraSession {
        <<MainActor, NSObject>>
        +previewLayer : AVCaptureVideoPreviewLayer
        +framePublisher : AnyPublisher~CVPixelBuffer~
        +start() throws
        +stop()
    }
    class PoseEstimator {
        <<class, Sendable>>
        +detectPoses(CVPixelBuffer) [VNHumanBodyPoseObservation]
    }
    class KeypointConverter {
        <<struct>>
        +oppOrder : [Int]$
        +coco17Joints : [JointName]$
        +convert(VNHumanBodyPoseObservation, Int, CMTime) PoseSkeleton
        +selectNeck(Keypoint?, Keypoint, Keypoint) Keypoint$
        +reorder([Keypoint], Keypoint) [Keypoint]$
    }
    class PoseNormalizer {
        <<struct>>
        +normalize([PoseSkeleton]) MLMultiArray
    }
    class AnomalyScorer {
        <<struct, Sendable>>
        +threshold : Float
        +classify(Float, Bool) AnomalyResult
    }
    class FrameBuffer {
        <<actor>>
        +capacity : Int = 24$
        +isReady : Bool
        +count : Int
        +append(PoseSkeleton)
        +currentWindow() [PoseSkeleton]?
        +reset()
    }
    class STGNFModelRunner {
        <<class, Sendable>>
        +runInference(MLMultiArray) Float
    }
    class STGNFModelProtocol {
        <<protocol>>
        +runInference(MLMultiArray) Float
    }

    %% ─── Domain models ───────────────────────────────────────────────────────
    class Keypoint {
        <<struct, Sendable>>
        +x : Float
        +y : Float
        +confidence : Float
    }
    class PoseSkeleton {
        <<struct, Sendable>>
        +keypoints : [Keypoint]
        +frameIndex : Int
        +timestamp : CMTime
        +boundingBox : CGRect
    }
    class AnomalyResult {
        <<struct, Sendable>>
        +score : Float
        +label : AnomalyLabel
        +timestamp : Date
    }
    class AnomalyLabel {
        <<enumeration>>
        normal
        anomaly
        warmup
    }
    class DetectionState {
        <<enumeration>>
        idle
        warmingUp(framesCollected, framesNeeded)
        running(latestResult)
        error(reason)
    }

    %% ─── App → View routing ──────────────────────────────────────────────────
    ShopliftDetectApp --> HomeView       : if onboarded
    ShopliftDetectApp --> OnboardingView : if not onboarded

    %% ─── View → ViewModel ownership ─────────────────────────────────────────
    HomeView       *-- HomeViewModel       : @StateObject
    OnboardingView *-- OnboardingViewModel : @StateObject
    DetectionView  *-- DetectionViewModel  : @StateObject
    PosePreviewView *-- PosePreviewViewModel : @StateObject

    %% ─── View navigation ─────────────────────────────────────────────────────
    HomeView --> DetectionView   : fullScreenCover
    HomeView --> PosePreviewView : fullScreenCover
    OnboardingView --> OnboardingPageView : contains pages

    %% ─── View composition ────────────────────────────────────────────────────
    DetectionView  --> SkeletonOverlayView : renders
    DetectionView  --> ScoreCardView       : renders
    DetectionView  --> CameraPreviewLayer  : renders
    PosePreviewView --> SkeletonOverlayView : renders
    PosePreviewView --> CameraPreviewLayer  : renders

    %% ─── ViewModel → Service ownership ──────────────────────────────────────
    DetectionViewModel  *-- CameraSession     : owns
    DetectionViewModel  *-- PoseEstimator     : owns
    DetectionViewModel  *-- KeypointConverter : owns
    DetectionViewModel  *-- AnomalyScorer     : owns
    DetectionViewModel  *-- STGNFModelRunner  : owns (lazy)
    DetectionViewModel "1" *-- "0..*" FrameBuffer : trackBuffers[trackID]

    PosePreviewViewModel *-- CameraSession     : owns
    PosePreviewViewModel *-- PoseEstimator     : owns
    PosePreviewViewModel *-- KeypointConverter : owns

    %% ─── Service → Domain ────────────────────────────────────────────────────
    KeypointConverter ..> PoseSkeleton  : produces
    PoseSkeleton       *-- Keypoint     : 18 keypoints
    AnomalyScorer      ..> AnomalyResult : produces
    AnomalyResult      *-- AnomalyLabel  : label
    DetectionState     ..> AnomalyResult : running case
    FrameBuffer        *-- PoseSkeleton  : rolling window
    PoseNormalizer     ..> PoseSkeleton  : consumes
    DetectionViewModel ..> PoseNormalizer : uses per window

    %% ─── Protocol conformance ────────────────────────────────────────────────
    STGNFModelRunner ..|> STGNFModelProtocol
```
