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

    func testWarmupIndicatorVisibleOnLaunch() {
        app.buttons["startDetectionButton"].tap()
        XCTAssertTrue(app.staticTexts["warmupIndicator"].waitForExistence(timeout: 5))
    }

    func testDismissReturnsToHome() {
        app.buttons["startDetectionButton"].tap()
        let dismiss = app.buttons["xmark.circle.fill"]
        XCTAssertTrue(dismiss.waitForExistence(timeout: 3))
        dismiss.tap()
        XCTAssertTrue(app.buttons["startDetectionButton"].waitForExistence(timeout: 3))
    }
}
