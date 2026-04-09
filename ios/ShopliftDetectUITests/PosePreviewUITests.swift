import XCTest

final class PosePreviewUITests: XCTestCase {

    private var app: XCUIApplication!

    override func setUp() {
        super.setUp()
        continueAfterFailure = false
        app = XCUIApplication()
        app.launchArguments = ["--skip-onboarding", "--ui-test-pose-preview"]
        app.launch()
    }

    func testPosePreviewDismissButtonExists() {
        app.buttons["posePreviewButton"].tap()
        XCTAssertTrue(app.buttons["posePreviewDismissButton"].waitForExistence(timeout: 3))
    }

    func testPosePreviewShowsSkeletonCountBadge() {
        app.buttons["posePreviewButton"].tap()
        XCTAssertTrue(app.staticTexts["posePreviewCount"].waitForExistence(timeout: 3))
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
}
