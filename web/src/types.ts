export type DeviceRole = "camera" | "supervisor";

export type ConnectionState =
  | "connected"
  | "listening"
  | "connecting"
  | "handshaking"
  | "stale"
  | "disconnected"
  | "failed"
  | "idle";

export type AnomalyLabel = "normal" | "anomaly" | "warmup";

export type Appearance = "system" | "light" | "dark";

export interface PairingPayload {
  host: string;
  port: number;
  token: string;
}

export interface PairingSession {
  sessionID: string;
  role: "supervisor";
  deviceName: string;
  host: string;
  port: number;
  connectionState: ConnectionState;
  token: string;
  createdAt: string;
}

export interface DetectionResult {
  trackID: number;
  score: number;
  label: AnomalyLabel;
  timestamp: string;
}

export interface VideoFrameSummary {
  timestamp: number;
  width: number;
  height: number;
  imageDataUrl?: string;
}

export interface SupervisorFeedTileState {
  sessionID: string;
  deviceName: string;
  host: string;
  port: number;
  connectionState: ConnectionState;
  latestFrame: VideoFrameSummary | null;
  latestDetections: DetectionResult[];
}

export interface WebSupervisorState {
  onboardingComplete: boolean;
  selectedRole: DeviceRole | null;
  appearance: Appearance;
  sessions: PairingSession[];
}
