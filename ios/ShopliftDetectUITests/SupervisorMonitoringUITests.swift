import XCTest

final class SupervisorMonitoringUITests: XCTestCase {
    private var app: XCUIApplication!

    override func setUp() {
        super.setUp()
        continueAfterFailure = false
        app = XCUIApplication()
    }

    func testEmptySupervisorGridShowsPlaceholder() {
        app.launchArguments = ["--skip-onboarding", "--ui-test-supervisor-role", "--ui-test-supervisor-capable-device"]
        app.launch()

        XCTAssertTrue(app.staticTexts["supervisorEmptyStateLabel"].waitForExistence(timeout: 3))
    }

    func testSupervisorGridCanShowLiveTileAndOpenDetail() {
        app.launchArguments = ["--skip-onboarding", "--ui-test-supervisor-role", "--ui-test-supervisor-feed", "--ui-test-supervisor-capable-device"]
        app.launch()

        let tile = app.buttons["supervisorTile_Aisle 3 Camera"]
        XCTAssertTrue(tile.waitForExistence(timeout: 3))
        tile.tap()

        XCTAssertTrue(app.staticTexts["cameraFeedDetailTitle"].waitForExistence(timeout: 3))
    }

    func testFifthCameraAttemptShowsLimitError() {
        app.launchArguments = ["--skip-onboarding", "--ui-test-supervisor-role", "--ui-test-supervisor-maxed", "--ui-test-supervisor-capable-device"]
        app.launch()

        XCTAssertTrue(app.staticTexts["supervisorLimitBanner"].waitForExistence(timeout: 3))
    }

    func testDarkModeCarriesIntoSupervisorDetail() {
        app.launchArguments = [
            "--skip-onboarding",
            "--ui-test-supervisor-role",
            "--ui-test-supervisor-feed",
            "--ui-test-supervisor-capable-device",
            "--ui-test-dark-mode"
        ]
        app.launch()

        XCTAssertTrue(app.otherElements["supervisorScreen"].waitForExistence(timeout: 3))
        XCTAssertEqual(app.otherElements["supervisorScreen"].value as? String, "dark")

        let tile = app.buttons["supervisorTile_Aisle 3 Camera"]
        XCTAssertTrue(tile.waitForExistence(timeout: 3))
        tile.tap()

        XCTAssertTrue(app.otherElements["cameraFeedDetailScreen"].waitForExistence(timeout: 3))
        XCTAssertEqual(app.otherElements["cameraFeedDetailScreen"].value as? String, "dark")
    }
}
