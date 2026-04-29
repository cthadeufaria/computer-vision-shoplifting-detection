import { describe, expect, it } from "vitest";
import { blockedRoleReason, canSelectWebRole } from "./capabilities";

describe("web role capabilities", () => {
  it("allows the supervisor role", () => {
    expect(canSelectWebRole("supervisor")).toBe(true);
    expect(blockedRoleReason("supervisor")).toBeNull();
  });

  it("blocks the smart camera role on web", () => {
    expect(canSelectWebRole("camera")).toBe(false);
    expect(blockedRoleReason("camera")).toContain("native iOS app");
  });
});
