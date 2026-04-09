import XCTest

final class OnboardingUITests: XCTestCase {

    private var app: XCUIApplication!

    override func setUp() {
        super.setUp()
        continueAfterFailure = false
        app = XCUIApplication()
        app.launchArguments = ["--reset-onboarding", "--ui-test-camera-authorized"]
        app.launch()
    }

    func testFirstScreenIsWelcome() {
        XCTAssertTrue(app.staticTexts["Welcome"].waitForExistence(timeout: 3))
    }

    func testNextButtonAdvancesToPageTwo() {
        let next = app.buttons["nextButton"]
        XCTAssertTrue(next.waitForExistence(timeout: 3))
        next.tap()
        XCTAssertTrue(app.staticTexts["How It Works"].waitForExistence(timeout: 3))
    }

    func testThirdScreenHasPermissionCTA() {
        app.buttons["nextButton"].tap()
        app.buttons["nextButton"].tap()
        XCTAssertTrue(app.buttons["cameraRoleButton"].waitForExistence(timeout: 3))
    }

    func testCanCompleteCameraOnboardingFlow() {
        app.buttons["nextButton"].tap()
        app.buttons["nextButton"].tap()
        app.buttons["cameraRoleButton"].tap()
        app.buttons["nextButton"].tap()
        app.buttons["grantCameraAccessButton"].tap()

        XCTAssertTrue(app.buttons["startDetectionButton"].waitForExistence(timeout: 5))
    }

    func testSupervisorRoleRoutesToSupervisorHome() {
        app.buttons["nextButton"].tap()
        app.buttons["nextButton"].tap()
        app.buttons["supervisorRoleButton"].tap()
        app.buttons["nextButton"].tap()
        app.buttons["grantCameraAccessButton"].tap()

        XCTAssertTrue(app.staticTexts["supervisorHomeTitle"].waitForExistence(timeout: 5))
    }

    func testOnboardingSkippedOnSecondLaunch() {
        app.buttons["nextButton"].tap()
        app.buttons["nextButton"].tap()
        app.buttons["cameraRoleButton"].tap()
        app.buttons["nextButton"].tap()
        app.buttons["grantCameraAccessButton"].tap()
        XCTAssertTrue(app.buttons["startDetectionButton"].waitForExistence(timeout: 5))

        app.terminate()
        app.launchArguments = []
        app.launch()
        XCTAssertTrue(app.buttons["startDetectionButton"].waitForExistence(timeout: 3))
    }
}
