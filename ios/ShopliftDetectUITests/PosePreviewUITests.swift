import XCTest

final class PosePreviewUITests: XCTestCase {

    private var app: XCUIApplication!

    override func setUp() {
        super.setUp()
        continueAfterFailure = false
        app = XCUIApplication()
        app.launchArguments = ["--skip-onboarding", "--ui-test-pose-preview", "--ui-test-supervisor-capable-device"]
        app.launch()
    }

    func testPosePreviewDismissButtonExists() {
        app.buttons["posePreviewButton"].tap()
        XCTAssertTrue(app.buttons["posePreviewDismissButton"].waitForExistence(timeout: 3))
    }

    func testPosePreviewShowsSkeletonCountBadge() {
        app.buttons["posePreviewButton"].tap()
        let countBadge = app.descendants(matching: .any)["posePreviewCount"]
        XCTAssertTrue(countBadge.waitForExistence(timeout: 3))
    }

    func testPosePreviewShowsCameraFeed() {
        app.buttons["posePreviewButton"].tap()
        XCTAssertTrue(app.otherElements["cameraPreview"].waitForExistence(timeout: 3))
    }

    func testPosePreviewDismissReturnsToHome() {
        app.buttons["posePreviewButton"].tap()
        let dismiss = app.buttons["posePreviewDismissButton"]
        XCTAssertTrue(dismiss.waitForExistence(timeout: 3))
        let cameraErrorAlert = app.alerts["Camera Error"]

        if cameraErrorAlert.waitForExistence(timeout: 1) {
            cameraErrorAlert.buttons["OK"].tap()
        } else {
            dismiss.tap()
        }

        XCTAssertTrue(app.buttons["startDetectionButton"].waitForExistence(timeout: 3))
    }

    func testPosePreviewCanBeRepeatedlyPresentedAndDismissedBeforeStartingDetection() {
        for _ in 0..<3 {
            app.buttons["posePreviewButton"].tap()
            let dismiss = app.buttons["posePreviewDismissButton"]
            XCTAssertTrue(dismiss.waitForExistence(timeout: 3))
            dismiss.tap()
            XCTAssertTrue(app.buttons["startDetectionButton"].waitForExistence(timeout: 3))
        }

        app.terminate()
        app = XCUIApplication()
        app.launchArguments = ["--skip-onboarding", "--ui-test-supervisor-capable-device"]
        app.launch()
        app.buttons["startDetectionButton"].tap()

        XCTAssertTrue(app.buttons["dismissStreamingButton"].waitForExistence(timeout: 3))
    }

    func testDarkModeCarriesIntoPosePreview() {
        app.terminate()
        app = XCUIApplication()
        app.launchArguments = [
            "--skip-onboarding",
            "--ui-test-pose-preview",
            "--ui-test-supervisor-capable-device",
            "--ui-test-dark-mode"
        ]
        app.launch()

        app.buttons["posePreviewButton"].tap()

        XCTAssertTrue(app.otherElements["posePreviewScreen"].waitForExistence(timeout: 3))
        XCTAssertEqual(app.otherElements["posePreviewScreen"].value as? String, "dark")
    }
}
