# Contract: Pairing Protocol

## Purpose

Defines the QR payload and authenticated JSON handshake between a camera device and a supervisory device.

## QR Payload

- **Format**: `sdlink://<lan-host>:<port>?token=<single-use-token>`
- **Example**: `sdlink://192.168.1.24:7890?token=4F8A1D2C`

## Validation Rules

- Scheme must be `sdlink`
- Host must be a LAN-reachable IP or hostname for the camera device
- Port must be a valid listener port for the camera device
- `token` must be present
- Token is valid only while the camera pairing screen is visible

## Handshake Flow

### 1. Supervisor opens connection

- Transport: `NWConnection` to `<lan-host>:<port>`
- Timeout budget: 5 seconds to receive handshake response

### 2. Supervisor sends `hello`

```json
{
  "type": "hello",
  "protocolVersion": 1,
  "role": "supervisor",
  "deviceName": "Front Desk iPad",
  "token": "4F8A1D2C"
}
```

### 3. Camera validates request

- Token present
- Token matches currently visible pairing token
- Token not expired, consumed, or reused
- Supervisor capacity does not exceed four feeds on the receiving supervisor side

### 4. Camera replies

Success:

```json
{
  "type": "hello_ack",
  "protocolVersion": 1,
  "role": "camera",
  "deviceName": "Aisle 3 Camera",
  "heartbeatIntervalMs": 2000,
  "streamFpsCap": 10
}
```

Failure:

```json
{
  "type": "hello_reject",
  "protocolVersion": 1,
  "reason": "invalid_token"
}
```

## Error Reasons

- `invalid_payload`
- `invalid_token`
- `expired_token`
- `reused_token`
- `unsupported_version`
- `connection_limit_reached`

## Session Rules

- Camera consumes the token after a successful `hello`
- Camera invalidates the token as soon as the pairing screen disappears
- Both peers send heartbeat messages at `heartbeatIntervalMs`
- Missing heartbeat/frame updates for 5 seconds transitions the supervisor tile to stale/disconnected
