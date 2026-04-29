import { useState } from "react";
import type { Appearance, DeviceRole } from "../types";
import { blockedRoleReason } from "../services/capabilities";
import RoleSelection from "./RoleSelection";

interface OnboardingProps {
  selectedRole: DeviceRole | null;
  appearance: Appearance;
  onSelectRole: (role: DeviceRole) => void;
  onSelectAppearance: (appearance: Appearance) => void;
  onComplete: () => void;
}

const pages = [
  {
    eyebrow: "Web Supervisor",
    title: "Welcome to ShopliftDetect",
    body: "Monitor paired smart camera feeds from a browser while keeping camera capture on the native iOS devices.",
    icon: "figure.walk"
  },
  {
    eyebrow: "How it works",
    title: "Supervision without browser-side capture",
    body: "Smart cameras stream pose-aware video summaries over the local network. The web view focuses on pairing, feed status, and anomaly review.",
    icon: "cpu"
  },
  {
    eyebrow: "Role",
    title: "Choose this device role",
    body: "The web target is supervisor-only. Smart Camera remains available in the iOS app where camera, CoreML, and local streaming APIs are present.",
    icon: "rectangle.grid.2x2"
  },
  {
    eyebrow: "Ready",
    title: "Supervisor setup",
    body: "Finish setup to open the web supervisor dashboard. Pairing can be done from the dashboard by pasting a camera sdlink payload.",
    icon: "qrcode.viewfinder"
  }
];

export default function Onboarding({
  selectedRole,
  appearance,
  onSelectRole,
  onSelectAppearance,
  onComplete
}: OnboardingProps) {
  const [page, setPage] = useState(0);
  const [notice, setNotice] = useState<string | null>(null);
  const current = pages[page];
  const isRolePage = page === 2;
  const isFinalPage = page === pages.length - 1;
  const canAdvance = !isRolePage || selectedRole === "supervisor";

  function handleRoleSelect(role: DeviceRole) {
    const blockedReason = blockedRoleReason(role);
    if (blockedReason) {
      setNotice(blockedReason);
      return;
    }

    setNotice(null);
    onSelectRole(role);
  }

  function next() {
    if (!canAdvance) {
      setNotice("Select Supervisory View to continue in the web app.");
      return;
    }

    setPage((currentPage) => Math.min(currentPage + 1, pages.length - 1));
  }

  return (
    <main className="onboarding-shell">
      <section className="hero-panel">
        <div className="brand-row">
          <span className="brand-mark" aria-hidden="true">
            SD
          </span>
          <span>ShopliftDetect</span>
        </div>
        <div className="page-indicator" aria-label={`Step ${page + 1} of ${pages.length}`}>
          {pages.map((item, index) => (
            <span key={item.title} className={index === page ? "active" : ""} />
          ))}
        </div>
        <div className="hero-card">
          <span className="eyebrow">{current.eyebrow}</span>
          <div className="symbol-tile" aria-hidden="true">
            {current.icon}
          </div>
          <h1>{current.title}</h1>
          <p>{current.body}</p>
        </div>
      </section>

      <section className="setup-panel" aria-label="Web supervisor onboarding">
        <AppearancePicker selected={appearance} onSelect={onSelectAppearance} />

        {isRolePage ? (
          <RoleSelection selectedRole={selectedRole} onSelectRole={handleRoleSelect} />
        ) : (
          <div className="setup-copy">
            <h2>{current.title}</h2>
            <p>{current.body}</p>
            {isFinalPage ? (
              <div className="permission-summary">
                <strong>Browser capability guard</strong>
                <span>Smart Camera setup is blocked on web. This target always enters Supervisory View.</span>
              </div>
            ) : null}
          </div>
        )}

        {notice ? (
          <div className="notice" role="status">
            {notice}
          </div>
        ) : null}

        <div className="onboarding-actions">
          <button className="secondary-button" type="button" disabled={page === 0} onClick={() => setPage((value) => value - 1)}>
            Back
          </button>
          {isFinalPage ? (
            <button className="primary-button" type="button" onClick={onComplete}>
              Open Supervisor Dashboard
            </button>
          ) : (
            <button className="primary-button" type="button" onClick={next} disabled={!canAdvance}>
              Next
            </button>
          )}
        </div>
      </section>
    </main>
  );
}

function AppearancePicker({
  selected,
  onSelect
}: {
  selected: Appearance;
  onSelect: (appearance: Appearance) => void;
}) {
  return (
    <div className="appearance-picker" aria-label="Appearance">
      {(["system", "light", "dark"] as const).map((appearance) => (
        <button
          key={appearance}
          className={selected === appearance ? "selected" : ""}
          type="button"
          onClick={() => onSelect(appearance)}
        >
          {appearance}
        </button>
      ))}
    </div>
  );
}
