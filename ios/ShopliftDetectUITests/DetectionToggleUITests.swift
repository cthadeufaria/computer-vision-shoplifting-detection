import XCTest

final class DetectionToggleUITests: XCTestCase {

    private var app: XCUIApplication!

    override func setUp() {
        super.setUp()
        continueAfterFailure = false
        launchApp()
    }

    private func launchApp(additionalArguments: [String] = []) {
        app = XCUIApplication()
        app.launchArguments = ["--skip-onboarding"] + additionalArguments
        app.launch()
    }

    func testHomeShowsStartDetectionButton() {
        XCTAssertTrue(app.buttons["startDetectionButton"].waitForExistence(timeout: 3))
    }

    func testStartDetectionPresentsDetectionView() {
        app.terminate()
        launchApp(additionalArguments: ["--ui-test-detection-preview"])

        app.buttons["startDetectionButton"].tap()
        XCTAssertTrue(app.buttons["xmark.circle.fill"].waitForExistence(timeout: 3))
    }

    func testPosePreviewPresentsPoseOnlyView() {
        app.buttons["posePreviewButton"].tap()
        XCTAssertTrue(app.buttons["posePreviewDismissButton"].waitForExistence(timeout: 3))
        XCTAssertTrue(app.otherElements["cameraPreview"].waitForExistence(timeout: 3))
    }

    func testWarmupIndicatorVisibleOnLaunch() {
        app.terminate()
        launchApp(additionalArguments: ["--ui-test-detection-preview"])

        app.buttons["startDetectionButton"].tap()
        XCTAssertTrue(app.staticTexts["warmupIndicator"].waitForExistence(timeout: 5))
    }

    func testDetectionViewShowsThresholdControls() {
        app.terminate()
        launchApp(additionalArguments: ["--ui-test-detection-preview"])

        app.buttons["startDetectionButton"].tap()

        XCTAssertTrue(app.staticTexts["thresholdValueLabel"].waitForExistence(timeout: 5))
        XCTAssertTrue(app.buttons["decreaseThresholdButton"].waitForExistence(timeout: 5))
        XCTAssertTrue(app.buttons["increaseThresholdButton"].waitForExistence(timeout: 5))
    }

    func testHomeShowsCurrentThreshold() {
        XCTAssertTrue(app.staticTexts["homeThresholdLabel"].waitForExistence(timeout: 3))
    }

    func testCameraPreviewRemainsVisibleDuringWarmup() {
        app.terminate()
        launchApp(additionalArguments: ["--ui-test-detection-preview"])

        app.buttons["startDetectionButton"].tap()

        let warmup = app.staticTexts["warmupIndicator"]
        let preview = app.otherElements["cameraPreview"]

        XCTAssertTrue(warmup.waitForExistence(timeout: 5))
        XCTAssertTrue(preview.waitForExistence(timeout: 5))
    }

    func testDismissReturnsToHome() {
        app.terminate()
        launchApp(additionalArguments: ["--ui-test-detection-preview"])

        app.buttons["startDetectionButton"].tap()
        let dismiss = app.buttons["xmark.circle.fill"]
        XCTAssertTrue(dismiss.waitForExistence(timeout: 3))
        dismiss.tap()
        XCTAssertTrue(app.buttons["startDetectionButton"].waitForExistence(timeout: 3))
    }
}
