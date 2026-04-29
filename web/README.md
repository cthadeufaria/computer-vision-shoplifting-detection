# ShopliftDetect Web Supervisor

This is a browser version of the supervisor flow from the native iOS app. It intentionally does not expose the Smart Camera path: web onboarding disables that role and the role-selection state guard rejects it.

## Run

```bash
cd web
npm install
npm run dev
```

Open the Vite URL in a browser. The dashboard can pair a camera by pasting an iOS Smart Camera payload such as:

```text
sdlink://192.168.1.24:7890?token=ABCD1234
```

## Local Wi-Fi MVP Flow

1. Put the iOS app in Smart Camera mode.
2. Tap `Start Streaming`.
3. Keep the iOS app open; it starts a local WebSocket listener on the pairing payload host and port.
4. In the web supervisor, paste the `sdlink://...` payload shown by the iOS app.
5. The web supervisor connects to `ws://<iphone-lan-ip>:<port>/?token=<token>` and renders incoming JPEG frames.

## Scope

- Supervisor onboarding and dashboard are implemented for web.
- Smart Camera is blocked on web because camera capture, CoreML inference, and local frame streaming remain native iOS capabilities.
- The web dashboard stores paired sessions in `localStorage`, reconnects while the camera stream is unavailable, and consumes the iOS Smart Camera WebSocket stream when both devices are on the same LAN.
- The MVP stream is token-gated `ws://` for local Wi-Fi development. A production deployment should move this to authenticated `wss://` with certificate handling.
