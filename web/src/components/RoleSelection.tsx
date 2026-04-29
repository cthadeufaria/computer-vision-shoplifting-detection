import type { DeviceRole } from "../types";
import { WEB_SUPERVISOR_CAPABILITIES } from "../services/capabilities";

interface RoleSelectionProps {
  selectedRole: DeviceRole | null;
  onSelectRole: (role: DeviceRole) => void;
}

export default function RoleSelection({ selectedRole, onSelectRole }: RoleSelectionProps) {
  return (
    <div className="role-selection">
      <p className="role-intro">
        Smart Camera captures and streams from iOS. Supervisory View monitors paired camera feeds from this browser.
      </p>

      <RoleButton
        title="Smart Camera"
        subtitle="Unavailable on web. Use the iOS app to capture video, compress frames, and stream over Wi-Fi."
        role="camera"
        selectedRole={selectedRole}
        disabled={!WEB_SUPERVISOR_CAPABILITIES.supportsCameraRole}
        dataTestID="cameraRoleButton"
        onSelectRole={onSelectRole}
      />

      <RoleButton
        title="Supervisory View"
        subtitle="Use this browser to pair camera feeds, monitor status, and review anomaly signals."
        role="supervisor"
        selectedRole={selectedRole}
        disabled={!WEB_SUPERVISOR_CAPABILITIES.supportsSupervisorRole}
        dataTestID="supervisorRoleButton"
        onSelectRole={onSelectRole}
      />

      <p className="availability-note">
        Web builds are supervisor-only because browser APIs do not provide the native camera streaming, CoreML, and
        local transport path used by Smart Camera mode.
      </p>
    </div>
  );
}

function RoleButton({
  title,
  subtitle,
  role,
  selectedRole,
  disabled,
  dataTestID,
  onSelectRole
}: {
  title: string;
  subtitle: string;
  role: DeviceRole;
  selectedRole: DeviceRole | null;
  disabled: boolean;
  dataTestID: string;
  onSelectRole: (role: DeviceRole) => void;
}) {
  return (
    <button
      className={`role-button ${selectedRole === role ? "selected" : ""} ${disabled ? "blocked" : ""}`}
      type="button"
      disabled={disabled}
      data-testid={dataTestID}
      onClick={() => onSelectRole(role)}
    >
      <span>
        <strong>{title}</strong>
        <small>{subtitle}</small>
      </span>
      <span className="role-status">{disabled ? "Blocked" : selectedRole === role ? "Selected" : "Available"}</span>
    </button>
  );
}
