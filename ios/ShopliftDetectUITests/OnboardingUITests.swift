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

    func testCameraRoleShowsPairingQRCodeOnFinalScreen() {
        app.buttons["nextButton"].tap()
        app.buttons["nextButton"].tap()
        app.buttons["cameraRoleButton"].tap()
        app.buttons["nextButton"].tap()

        XCTAssertTrue(app.otherElements["pairingQRCodeView"].waitForExistence(timeout: 3))
        XCTAssertTrue(app.staticTexts["pairingQRCodePayloadLabel"].waitForExistence(timeout: 3))
    }

    func testCanCompleteCameraOnboardingFlow() {
        app.buttons["nextButton"].tap()
        app.buttons["nextButton"].tap()
        app.buttons["cameraRoleButton"].tap()
        app.buttons["nextButton"].tap()
        app.buttons["grantCameraAccessButton"].tap()

        XCTAssertTrue(app.buttons["startDetectionButton"].waitForExistence(timeout: 5))
    }

    func testCameraOnlyDeviceDisablesSupervisorRole() {
        app.terminate()
        app.launchArguments = [
            "--reset-onboarding",
            "--ui-test-camera-authorized",
            "--ui-test-camera-only-device"
        ]
        app.launch()

        app.buttons["nextButton"].tap()
        app.buttons["nextButton"].tap()

        XCTAssertTrue(app.staticTexts["supervisorAvailabilityNote"].waitForExistence(timeout: 3))
        XCTAssertFalse(app.buttons["supervisorRoleButton"].isEnabled)
    }

    func testSupervisorRoleRoutesToSupervisorHome() {
        app.terminate()
        app.launchArguments = [
            "--reset-onboarding",
            "--ui-test-camera-authorized",
            "--ui-test-supervisor-capable-device",
            "--ui-test-pairing-payload=sdlink://192.168.1.24:7890?token=VALID123"
        ]
        app.launch()

        app.buttons["nextButton"].tap()
        app.buttons["nextButton"].tap()
        app.buttons["supervisorRoleButton"].tap()
        app.buttons["nextButton"].tap()
        app.buttons["scanQRCodeButton"].tap()
        app.buttons["grantCameraAccessButton"].tap()

        XCTAssertTrue(app.staticTexts["supervisorHomeTitle"].waitForExistence(timeout: 5))
    }

    func testSupervisorRoleCanScanPrefilledPayloadAndFinishSetup() {
        app.terminate()
        app.launchArguments = [
            "--reset-onboarding",
            "--ui-test-camera-authorized",
            "--ui-test-supervisor-capable-device",
            "--ui-test-pairing-payload=sdlink://192.168.1.24:7890?token=VALID123"
        ]
        app.launch()

        app.buttons["nextButton"].tap()
        app.buttons["nextButton"].tap()
        app.buttons["supervisorRoleButton"].tap()
        app.buttons["nextButton"].tap()
        app.buttons["scanQRCodeButton"].tap()
        app.buttons["grantCameraAccessButton"].tap()

        XCTAssertTrue(app.staticTexts["supervisorHomeTitle"].waitForExistence(timeout: 5))
        XCTAssertEqual(app.staticTexts["supervisorConnectionStatusLabel"].label, "Connected")
    }

    func testSupervisorRoleShowsErrorForInvalidTokenAndStaysOnOnboarding() {
        app.terminate()
        app.launchArguments = [
            "--reset-onboarding",
            "--ui-test-camera-authorized",
            "--ui-test-supervisor-capable-device",
            "--ui-test-pairing-payload=sdlink://192.168.1.24:7890?token=WRONG999",
            "--ui-test-required-token=VALID123"
        ]
        app.launch()

        app.buttons["nextButton"].tap()
        app.buttons["nextButton"].tap()
        app.buttons["supervisorRoleButton"].tap()
        app.buttons["nextButton"].tap()
        app.buttons["scanQRCodeButton"].tap()

        XCTAssertTrue(app.alerts["Setup Incomplete"].waitForExistence(timeout: 3))
        XCTAssertFalse(app.staticTexts["supervisorHomeTitle"].exists)
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

    func testSupervisorOnboardingSkippedOnSecondLaunch() {
        app.terminate()
        app.launchArguments = [
            "--reset-onboarding",
            "--ui-test-camera-authorized",
            "--ui-test-supervisor-capable-device",
            "--ui-test-pairing-payload=sdlink://192.168.1.24:7890?token=VALID123"
        ]
        app.launch()

        app.buttons["nextButton"].tap()
        app.buttons["nextButton"].tap()
        app.buttons["supervisorRoleButton"].tap()
        app.buttons["nextButton"].tap()
        app.buttons["scanQRCodeButton"].tap()
        app.buttons["grantCameraAccessButton"].tap()
        XCTAssertTrue(app.staticTexts["supervisorHomeTitle"].waitForExistence(timeout: 5))

        app.terminate()
        app.launchArguments = []
        app.launch()

        XCTAssertTrue(app.staticTexts["supervisorHomeTitle"].waitForExistence(timeout: 3))
    }
}
