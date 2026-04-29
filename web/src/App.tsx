import { useEffect, useState } from "react";
import Onboarding from "./components/Onboarding";
import SupervisorDashboard from "./components/SupervisorDashboard";
import type { Appearance, DeviceRole, PairingSession, WebSupervisorState } from "./types";
import { canSelectWebRole } from "./services/capabilities";
import {
  DEFAULT_WEB_SUPERVISOR_STATE,
  clearWebSupervisorState,
  loadWebSupervisorState,
  saveWebSupervisorState
} from "./services/storage";

export default function App() {
  const [state, setState] = useState<WebSupervisorState>(() => loadWebSupervisorState());

  useEffect(() => {
    document.documentElement.dataset.appearance = state.appearance;
    saveWebSupervisorState(state);
  }, [state]);

  function selectRole(role: DeviceRole) {
    if (!canSelectWebRole(role)) {
      return;
    }

    setState((current) => ({ ...current, selectedRole: role }));
  }

  function selectAppearance(appearance: Appearance) {
    setState((current) => ({ ...current, appearance }));
  }

  function completeOnboarding() {
    setState((current) => ({
      ...current,
      onboardingComplete: true,
      selectedRole: "supervisor"
    }));
  }

  function setSessions(sessions: PairingSession[]) {
    setState((current) => ({ ...current, sessions }));
  }

  function resetOnboarding() {
    clearWebSupervisorState();
    setState(DEFAULT_WEB_SUPERVISOR_STATE);
  }

  if (!state.onboardingComplete) {
    return (
      <Onboarding
        selectedRole={state.selectedRole}
        appearance={state.appearance}
        onSelectRole={selectRole}
        onSelectAppearance={selectAppearance}
        onComplete={completeOnboarding}
      />
    );
  }

  return (
    <SupervisorDashboard
      appearance={state.appearance}
      sessions={state.sessions}
      onSelectAppearance={selectAppearance}
      onSetSessions={setSessions}
      onResetOnboarding={resetOnboarding}
    />
  );
}
