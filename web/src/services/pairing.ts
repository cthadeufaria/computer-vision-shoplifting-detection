import type { PairingPayload, PairingSession } from "../types";
import { WEB_SUPERVISOR_CAPABILITIES } from "./capabilities";

export class PairingError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "PairingError";
  }
}

export function parsePairingPayload(rawValue: string): PairingPayload {
  const value = rawValue.trim();
  if (!value) {
    throw new PairingError("Paste a camera pairing payload before connecting.");
  }

  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new PairingError("The pairing payload is not a valid URL.");
  }

  if (url.protocol !== "sdlink:") {
    throw new PairingError("Pairing payloads must use the sdlink:// scheme.");
  }

  const host = url.hostname;
  const port = Number(url.port);
  const token = url.searchParams.get("token")?.trim();

  if (!host || !Number.isInteger(port) || port <= 0 || port > 65535 || !token) {
    throw new PairingError("The pairing payload is missing a LAN host, port, or token.");
  }

  if (!isLanReachableHost(host)) {
    throw new PairingError("Only local network camera hosts are accepted.");
  }

  return { host, port, token };
}

export function createSupervisorSession(
  payload: PairingPayload,
  deviceName: string,
  existingSessions: PairingSession[]
): PairingSession {
  if (existingSessions.length >= WEB_SUPERVISOR_CAPABILITIES.maxFeeds) {
    throw new PairingError("v1 supports up to four simultaneous feeds.");
  }

  if (existingSessions.some((session) => session.token === payload.token)) {
    throw new PairingError("That camera pairing token is already connected.");
  }

  return {
    sessionID: makeID(),
    role: "supervisor",
    deviceName: deviceName.trim() || `Camera ${existingSessions.length + 1}`,
    host: payload.host,
    port: payload.port,
    connectionState: "connected",
    token: payload.token,
    createdAt: new Date().toISOString()
  };
}

function isLanReachableHost(host: string): boolean {
  if (host.endsWith(".local")) {
    return true;
  }

  const octets = host.split(".").map((part) => Number(part));
  if (octets.length !== 4 || octets.some((part) => !Number.isInteger(part) || part < 0 || part > 255)) {
    return false;
  }

  const [first, second] = octets;
  return first === 10 || (first === 172 && second >= 16 && second <= 31) || (first === 192 && second === 168);
}

function makeID(): string {
  if ("crypto" in globalThis && typeof globalThis.crypto.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }

  return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}
