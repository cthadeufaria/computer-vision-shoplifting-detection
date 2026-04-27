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
        app.launchArguments = ["--skip-onboarding", "--ui-test-supervisor-capable-device"] + additionalArguments
        app.launch()
    }

    func testHomeShowsStartDetectionButton() {
        XCTAssertTrue(app.buttons["startDetectionButton"].waitForExistence(timeout: 3))
    }

    func testStartStreamingPresentsStreamingView() {
        app.buttons["startDetectionButton"].tap()
        XCTAssertTrue(app.buttons["dismissStreamingButton"].waitForExistence(timeout: 3))
        XCTAssertTrue(app.staticTexts["streamingStatusLabel"].waitForExistence(timeout: 3))
    }

    func testPosePreviewPresentsPoseOnlyView() {
        app.buttons["posePreviewButton"].tap()
        XCTAssertTrue(app.buttons["posePreviewDismissButton"].waitForExistence(timeout: 3))
        XCTAssertTrue(app.otherElements["cameraPreview"].waitForExistence(timeout: 3))
    }

    func testStreamingViewShowsCameraPreview() {
        app.terminate()
        launchApp()

        app.buttons["startDetectionButton"].tap()
        XCTAssertTrue(app.otherElements["cameraPreview"].waitForExistence(timeout: 5))
    }

    func testHomeShowsStreamingDescription() {
        XCTAssertTrue(app.staticTexts["homeThresholdLabel"].waitForExistence(timeout: 3))
    }

    func testCameraOnlyDeviceHidesPosePreview() {
        app.terminate()
        launchApp(additionalArguments: ["--ui-test-camera-only-device"])

        XCTAssertFalse(app.buttons["posePreviewButton"].exists)
    }

    func testDismissStreamingReturnsToHome() {
        app.terminate()
        launchApp()

        app.buttons["startDetectionButton"].tap()
        let dismiss = app.buttons["dismissStreamingButton"]
        XCTAssertTrue(dismiss.waitForExistence(timeout: 3))
        let cameraErrorAlert = app.alerts["Camera Error"]

        if cameraErrorAlert.waitForExistence(timeout: 1) {
            cameraErrorAlert.buttons["OK"].tap()
        } else {
            dismiss.tap()
        }

        XCTAssertTrue(app.buttons["startDetectionButton"].waitForExistence(timeout: 3))
    }

    func testDarkModeCarriesIntoStreamingPresentation() {
        app.terminate()
        launchApp(additionalArguments: ["--ui-test-dark-mode"])

        XCTAssertTrue(app.otherElements["homeScreen"].waitForExistence(timeout: 3))
        XCTAssertEqual(app.otherElements["homeScreen"].value as? String, "dark")

        app.buttons["startDetectionButton"].tap()

        XCTAssertTrue(app.otherElements["cameraStreamingScreen"].waitForExistence(timeout: 3))
        XCTAssertEqual(app.otherElements["cameraStreamingScreen"].value as? String, "dark")
    }
}
