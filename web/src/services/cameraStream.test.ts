import { describe, expect, it } from "vitest";
import type { PairingSession } from "../types";
import { makeCameraStreamURL, parseCameraStreamMessage } from "./cameraStream";

const session: PairingSession = {
  sessionID: "session-1",
  role: "supervisor",
  deviceName: "Aisle Camera",
  host: "192.168.1.24",
  port: 7890,
  connectionState: "connected",
  token: "ABCD1234",
  createdAt: "2026-04-27T00:00:00.000Z"
};

describe("camera stream WebSocket helpers", () => {
  it("builds a token-authenticated LAN WebSocket URL", () => {
    expect(makeCameraStreamURL(session)).toBe("ws://192.168.1.24:7890/?token=ABCD1234");
  });

  it("parses video frame messages into renderable data URLs", () => {
    expect(
      parseCameraStreamMessage(
        JSON.stringify({
          type: "videoFrame",
          timestamp: 177000,
          width: 640,
          height: 360,
          jpegData: "AQID"
        })
      )
    ).toEqual({
      timestamp: 177000,
      width: 640,
      height: 360,
      imageDataUrl: "data:image/jpeg;base64,AQID"
    });
  });

  it("ignores unknown messages", () => {
    expect(parseCameraStreamMessage(JSON.stringify({ type: "heartbeat" }))).toBeNull();
  });
});
