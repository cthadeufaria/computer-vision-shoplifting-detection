import XCTest

final class DetectionToggleUITests: XCTestCase {

    private var app: XCUIApplication!

    override func setUp() {
        super.setUp()
        continueAfterFailure = false
        app = XCUIApplication()
        // Skip onboarding so we start directly at Home.
        app.launchArguments = ["--skip-onboarding"]
        app.launch()
    }

    func testHomeShowsStartDetectionButton() {
        XCTAssertTrue(app.buttons["startDetectionButton"].waitForExistence(timeout: 3))
    }

    func testStartDetectionPresentsDetectionView() {
        app.buttons["startDetectionButton"].tap()
        XCTAssertTrue(app.buttons["xmark.circle.fill"].waitForExistence(timeout: 3))
    }

    func testPosePreviewPresentsPoseOnlyView() {
        app.buttons["posePreviewButton"].tap()
        XCTAssertTrue(app.buttons["posePreviewDismissButton"].waitForExistence(timeout: 3))
        XCTAssertTrue(app.otherElements["cameraPreview"].waitForExistence(timeout: 3))
    }

    func testWarmupIndicatorVisibleOnLaunch() {
        app.buttons["startDetectionButton"].tap()
        XCTAssertTrue(app.staticTexts["warmupIndicator"].waitForExistence(timeout: 5))
    }

    func testCameraPreviewRemainsVisibleDuringWarmup() {
        app.terminate()
        app = XCUIApplication()
        app.launchArguments = ["--skip-onboarding", "--ui-test-detection-preview"]
        app.launch()

        app.buttons["startDetectionButton"].tap()

        let warmup = app.staticTexts["warmupIndicator"]
        let preview = app.otherElements["cameraPreview"]

        XCTAssertTrue(warmup.waitForExistence(timeout: 5))
        XCTAssertTrue(preview.waitForExistence(timeout: 5))
    }

    func testDismissReturnsToHome() {
        app.buttons["startDetectionButton"].tap()
        let dismiss = app.buttons["xmark.circle.fill"]
        XCTAssertTrue(dismiss.waitForExistence(timeout: 3))
        dismiss.tap()
        XCTAssertTrue(app.buttons["startDetectionButton"].waitForExistence(timeout: 3))
    }
}
