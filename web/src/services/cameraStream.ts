import type { PairingSession, VideoFrameSummary } from "../types";

export interface CameraStreamVideoFrameMessage {
  type: "videoFrame";
  timestamp: number;
  width: number;
  height: number;
  jpegData: string;
}

export function makeCameraStreamURL(session: PairingSession): string {
  const url = new URL(`ws://${session.host}:${session.port}/`);
  url.searchParams.set("token", session.token);
  return url.toString();
}

export function parseCameraStreamMessage(rawValue: string): VideoFrameSummary | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(rawValue);
  } catch {
    return null;
  }

  if (!isVideoFrameMessage(parsed)) {
    return null;
  }

  return {
    timestamp: parsed.timestamp,
    width: parsed.width,
    height: parsed.height,
    imageDataUrl: `data:image/jpeg;base64,${parsed.jpegData}`
  };
}

function isVideoFrameMessage(value: unknown): value is CameraStreamVideoFrameMessage {
  if (!value || typeof value !== "object") {
    return false;
  }

  const candidate = value as Record<string, unknown>;
  return (
    candidate.type === "videoFrame" &&
    typeof candidate.timestamp === "number" &&
    typeof candidate.width === "number" &&
    typeof candidate.height === "number" &&
    typeof candidate.jpegData === "string" &&
    candidate.jpegData.length > 0
  );
}
