import XCTest

final class OnboardingUITests: XCTestCase {

    private var app: XCUIApplication!

    override func setUp() {
        super.setUp()
        continueAfterFailure = false
        app = XCUIApplication()
        // Reset onboarding state for each test.
        app.launchArguments = ["--reset-onboarding"]
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
        XCTAssertTrue(app.buttons["grantCameraAccessButton"].waitForExistence(timeout: 3))
    }

    func testCanCompleteOnboardingFlow() {
        app.buttons["nextButton"].tap()
        app.buttons["nextButton"].tap()
        app.buttons["grantCameraAccessButton"].tap()
        // After granting permission (or dismissing permission dialog), Home screen should appear.
        XCTAssertTrue(app.buttons["startDetectionButton"].waitForExistence(timeout: 5))
    }

    func testOnboardingSkippedOnSecondLaunch() {
        // Complete onboarding first.
        app.buttons["nextButton"].tap()
        app.buttons["nextButton"].tap()
        app.buttons["grantCameraAccessButton"].tap()
        XCTAssertTrue(app.buttons["startDetectionButton"].waitForExistence(timeout: 5))

        // Relaunch without reset — should go straight to Home.
        app.terminate()
        app.launchArguments = []
        app.launch()
        XCTAssertTrue(app.buttons["startDetectionButton"].waitForExistence(timeout: 3))
    }
}
