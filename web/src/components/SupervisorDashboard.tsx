import { useEffect, useState } from "react";
import type { Appearance, ConnectionState, PairingSession, SupervisorFeedTileState, VideoFrameSummary } from "../types";
import { WEB_SUPERVISOR_CAPABILITIES } from "../services/capabilities";
import { makeCameraStreamURL, parseCameraStreamMessage } from "../services/cameraStream";
import {
  PairingError,
  createSupervisorSession,
  parsePairingPayload
} from "../services/pairing";

interface SupervisorDashboardProps {
  appearance: Appearance;
  sessions: PairingSession[];
  onSelectAppearance: (appearance: Appearance) => void;
  onSetSessions: (sessions: PairingSession[]) => void;
  onResetOnboarding: () => void;
}

interface LiveFeedState {
  connectionState: ConnectionState;
  latestFrame: VideoFrameSummary | null;
}

export default function SupervisorDashboard({
  appearance,
  sessions,
  onSelectAppearance,
  onSetSessions,
  onResetOnboarding
}: SupervisorDashboardProps) {
  const [payload, setPayload] = useState("");
  const [deviceName, setDeviceName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [selectedTile, setSelectedTile] = useState<SupervisorFeedTileState | null>(null);
  const [liveFeeds, setLiveFeeds] = useState<Record<string, LiveFeedState>>({});
  const tiles = sessions.slice(0, WEB_SUPERVISOR_CAPABILITIES.maxFeeds).map((session) =>
    makeLiveFeedTile(session, liveFeeds[session.sessionID])
  );
  const activeSelectedTile = selectedTile
    ? tiles.find((tile) => tile.sessionID === selectedTile.sessionID) ?? selectedTile
    : null;
  const connectedFeedCount = Object.values(liveFeeds).filter((feed) => feed.connectionState === "connected").length;
  const connectionStatusText =
    sessions.length === 0 ? "Not Paired" : connectedFeedCount > 0 ? "Connected" : "Connecting";

  useEffect(() => {
    const sockets: WebSocket[] = [];
    const retryTimers: number[] = [];
    let disposed = false;

    function updateConnectionState(sessionID: string, connectionState: ConnectionState) {
      setLiveFeeds((current) => ({
        ...current,
        [sessionID]: {
          connectionState,
          latestFrame: current[sessionID]?.latestFrame ?? null
        }
      }));
    }

    function updateLatestFrame(sessionID: string, latestFrame: VideoFrameSummary) {
      setLiveFeeds((current) => ({
        ...current,
        [sessionID]: {
          connectionState: "connected",
          latestFrame
        }
      }));
    }

    function connect(session: PairingSession) {
      if (disposed) {
        return;
      }

      updateConnectionState(session.sessionID, "connecting");

      const socket = new WebSocket(makeCameraStreamURL(session));
      sockets.push(socket);

      socket.addEventListener("open", () => {
        updateConnectionState(session.sessionID, "connected");
      });

      socket.addEventListener("message", (event) => {
        if (typeof event.data !== "string") {
          return;
        }

        const latestFrame = parseCameraStreamMessage(event.data);
        if (!latestFrame) {
          return;
        }

        updateLatestFrame(session.sessionID, latestFrame);
      });

      socket.addEventListener("close", () => {
        if (disposed) {
          return;
        }

        updateConnectionState(session.sessionID, "disconnected");

        retryTimers.push(window.setTimeout(() => connect(session), 2000));
      });

      socket.addEventListener("error", () => {
        updateConnectionState(session.sessionID, "failed");
      });
    }

    sessions.forEach(connect);

    setLiveFeeds((current) =>
      Object.fromEntries(Object.entries(current).filter(([sessionID]) => sessions.some((session) => session.sessionID === sessionID)))
    );

    return () => {
      disposed = true;
      retryTimers.forEach(window.clearTimeout);
      sockets.forEach((socket) => socket.close());
    };
  }, [sessions]);

  function pairCamera() {
    setError(null);

    try {
      const parsed = parsePairingPayload(payload);
      const session = createSupervisorSession(parsed, deviceName, sessions);
      onSetSessions([...sessions, session]);
      setPayload("");
      setDeviceName("");
    } catch (caught) {
      setError(caught instanceof PairingError ? caught.message : "Pairing failed.");
    }
  }

  function removeSession(sessionID: string) {
    onSetSessions(sessions.filter((session) => session.sessionID !== sessionID));
    setSelectedTile(null);
  }

  return (
    <main className="dashboard-shell">
      <header className="dashboard-header">
        <div>
          <span className="eyebrow">Web Supervisor</span>
          <h1>Supervisor Mode</h1>
          <p>Monitor paired smart camera feeds from this browser.</p>
        </div>
        <div className="header-actions">
          <AppearancePicker selected={appearance} onSelect={onSelectAppearance} />
          <button className="secondary-button compact" type="button" onClick={onResetOnboarding}>
            Reset Setup
          </button>
        </div>
      </header>

      <section className="status-strip" aria-label="Supervisor status">
        <StatusCard label="Pairing" value={connectionStatusText} />
        <StatusCard label="Feeds" value={`${sessions.length}/${WEB_SUPERVISOR_CAPABILITIES.maxFeeds}`} />
        <StatusCard label="Mode" value="Supervisor Only" />
      </section>

      <section className="pairing-panel" aria-label="Pair a camera">
        <div>
          <h2>Pair Camera</h2>
          <p>Paste the camera payload shown by the iOS Smart Camera setup or streaming screen.</p>
        </div>
        <div className="pairing-form">
          <label>
            Device name
            <input
              value={deviceName}
              placeholder="Aisle 3 Camera"
              onChange={(event) => setDeviceName(event.target.value)}
            />
          </label>
          <label>
            Pairing payload
            <input
              value={payload}
              placeholder="sdlink://192.168.1.24:7890?token=ABCD1234"
              onChange={(event) => setPayload(event.target.value)}
            />
          </label>
          <div className="pairing-actions">
            <button className="primary-button" type="button" onClick={pairCamera} disabled={sessions.length >= 4}>
              Pair Camera
            </button>
          </div>
          {error ? (
            <div className="notice error" role="alert">
              {error}
            </div>
          ) : null}
        </div>
      </section>

      {sessions.length >= WEB_SUPERVISOR_CAPABILITIES.maxFeeds ? (
        <div className="limit-banner">v1 supports up to four simultaneous feeds.</div>
      ) : null}

      <section className="feed-section" aria-label="Supervisor feeds">
        {tiles.length === 0 ? (
          <div className="empty-state">
            <h2>Pair a camera device to begin monitoring live feeds.</h2>
            <p>The web build blocks Smart Camera setup, so capture devices must be paired from iOS.</p>
          </div>
        ) : (
          <div className="feed-grid">
            {tiles.map((tile) => (
              <FeedTile key={tile.sessionID} tile={tile} onSelect={() => setSelectedTile(tile)} />
            ))}
          </div>
        )}
      </section>

      {activeSelectedTile ? (
        <FeedDetail
          tile={activeSelectedTile}
          onDismiss={() => setSelectedTile(null)}
          onRemove={() => removeSession(activeSelectedTile.sessionID)}
        />
      ) : null}
    </main>
  );
}

function StatusCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="status-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function FeedTile({ tile, onSelect }: { tile: SupervisorFeedTileState; onSelect: () => void }) {
  const latest = tile.latestDetections.at(-1);
  const badgeText = latest
    ? latest.label === "anomaly"
      ? "ANOMALY"
      : latest.label === "warmup"
        ? "WARMING UP"
        : "GOOD"
    : null;

  return (
    <button className="feed-tile" type="button" onClick={onSelect} data-testid={`supervisorTile_${tile.deviceName}`}>
      <div className="video-card">
        {tile.latestFrame?.imageDataUrl ? (
          <img src={tile.latestFrame.imageDataUrl} alt={`${tile.deviceName} live camera frame`} />
        ) : (
          <>
            <span className="video-glyph" aria-hidden="true">
              {tile.connectionState === "connected" ? "video.fill" : "video.slash"}
            </span>
            <strong>{statusText(tile.connectionState)}</strong>
          </>
        )}
      </div>
      <div className="tile-meta">
        <span>{tile.deviceName}</span>
        <small>
          {tile.host}:{tile.port}
        </small>
      </div>
      {badgeText ? <span className={`anomaly-badge ${latest?.label ?? "normal"}`}>{badgeText}</span> : null}
    </button>
  );
}

function FeedDetail({
  tile,
  onDismiss,
  onRemove
}: {
  tile: SupervisorFeedTileState;
  onDismiss: () => void;
  onRemove: () => void;
}) {
  const latest = tile.latestDetections.at(-1);

  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="feed-detail-title">
      <section className="feed-detail">
        <button className="close-button" type="button" onClick={onDismiss}>
          Done
        </button>
        <div className="detail-video">
          {tile.latestFrame?.imageDataUrl ? (
            <img src={tile.latestFrame.imageDataUrl} alt={`${tile.deviceName} live camera frame`} />
          ) : (
            <>
              <span aria-hidden="true">{tile.connectionState === "connected" ? "video.fill" : "video.slash"}</span>
              <strong>{statusText(tile.connectionState)}</strong>
            </>
          )}
        </div>
        <h2 id="feed-detail-title">{tile.deviceName}</h2>
        <p>
          {tile.host}:{tile.port}
        </p>
        {latest ? (
          <div className={`detail-badge ${latest.label}`}>
            <span>{latest.label === "anomaly" ? "ANOMALY" : latest.label === "warmup" ? "WARMING UP" : "GOOD"}</span>
            <small>score {latest.score.toFixed(2)}</small>
          </div>
        ) : null}
        <button className="danger-button" type="button" onClick={onRemove}>
          Remove Feed
        </button>
      </section>
    </div>
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
    <div className="appearance-picker compact-picker" aria-label="Appearance">
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

function statusText(state: string): string {
  switch (state) {
    case "connected":
      return "Connected";
    case "stale":
      return "Stale";
    case "disconnected":
      return "Disconnected";
    case "failed":
      return "Failed";
    case "connecting":
    case "handshaking":
      return "Connecting";
    case "listening":
      return "Listening";
    default:
      return "Idle";
  }
}

function makeLiveFeedTile(session: PairingSession, liveFeed?: LiveFeedState): SupervisorFeedTileState {
  return {
    sessionID: session.sessionID,
    deviceName: session.deviceName,
    host: session.host,
    port: session.port,
    connectionState: liveFeed?.connectionState ?? "connecting",
    latestFrame: liveFeed?.latestFrame ?? null,
    latestDetections: []
  };
}
