import { describe, expect, it } from "vitest";
import { createSupervisorSession, parsePairingPayload, PairingError } from "./pairing";

describe("pairing payload parsing", () => {
  it("accepts iOS camera sdlink payloads", () => {
    expect(parsePairingPayload("sdlink://192.168.1.24:7890?token=ABCD1234")).toEqual({
      host: "192.168.1.24",
      port: 7890,
      token: "ABCD1234"
    });
  });

  it("rejects non-local hosts", () => {
    expect(() => parsePairingPayload("sdlink://example.com:7890?token=ABCD1234")).toThrow(PairingError);
  });

  it("rejects malformed local IP addresses", () => {
    expect(() => parsePairingPayload("sdlink://192.168.1.999:7890?token=ABCD1234")).toThrow(PairingError);
  });

  it("enforces the supervisor feed limit", () => {
    const payload = parsePairingPayload("sdlink://192.168.1.24:7890?token=FIFTH");
    const existing = Array.from({ length: 4 }, (_, index) =>
      createSupervisorSession(
        parsePairingPayload(`sdlink://192.168.1.${20 + index}:7890?token=TOKEN${index}`),
        `Camera ${index + 1}`,
        []
      )
    );

    expect(() => createSupervisorSession(payload, "Camera 5", existing)).toThrow("four simultaneous feeds");
  });
});
