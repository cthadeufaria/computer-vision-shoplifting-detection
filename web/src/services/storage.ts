import type { WebSupervisorState } from "../types";

const STORAGE_KEY = "shoplift-detect-web-supervisor-state-v1";

export const DEFAULT_WEB_SUPERVISOR_STATE: WebSupervisorState = {
  onboardingComplete: false,
  selectedRole: null,
  appearance: "system",
  sessions: []
};

export function loadWebSupervisorState(): WebSupervisorState {
  if (typeof localStorage === "undefined") {
    return DEFAULT_WEB_SUPERVISOR_STATE;
  }

  const rawValue = localStorage.getItem(STORAGE_KEY);
  if (!rawValue) {
    return DEFAULT_WEB_SUPERVISOR_STATE;
  }

  try {
    const parsed = JSON.parse(rawValue) as Partial<WebSupervisorState>;
    return {
      onboardingComplete: Boolean(parsed.onboardingComplete),
      selectedRole: parsed.selectedRole === "supervisor" ? "supervisor" : null,
      appearance:
        parsed.appearance === "light" || parsed.appearance === "dark" || parsed.appearance === "system"
          ? parsed.appearance
          : "system",
      sessions: Array.isArray(parsed.sessions) ? parsed.sessions : []
    };
  } catch {
    return DEFAULT_WEB_SUPERVISOR_STATE;
  }
}

export function saveWebSupervisorState(state: WebSupervisorState): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

export function clearWebSupervisorState(): void {
  localStorage.removeItem(STORAGE_KEY);
}
