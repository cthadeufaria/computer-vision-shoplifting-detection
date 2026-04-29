import type { DeviceRole } from "../types";

export const WEB_SUPERVISOR_CAPABILITIES = {
  supportsCameraRole: false,
  supportsSupervisorRole: true,
  maxFeeds: 4
} as const;

export function canSelectWebRole(role: DeviceRole): boolean {
  return role === "supervisor";
}

export function blockedRoleReason(role: DeviceRole): string | null {
  if (canSelectWebRole(role)) {
    return null;
  }

  return "Smart Camera capture and frame streaming are only available in the native iOS app.";
}
