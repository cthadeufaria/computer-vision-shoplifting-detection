# Contract: Stream Protocol

## Purpose

Defines the framed transport used after pairing to deliver heartbeats, video frames, and detection results from camera to supervisor.

## Framing

- Connection: established after successful pairing handshake
- Transport: local-LAN peer connection managed by the networking service
- Envelope: length-prefixed binary frames

### Envelope Layout

1. `messageType` - 1 byte
2. `payloadLength` - 4 bytes unsigned big-endian
3. `payload` - `payloadLength` bytes

## Message Types

- `0x01` - Heartbeat
- `0x02` - Video frame
- `0x03` - Detection results
- `0x04` - Disconnect notice

## Heartbeat Payload

JSON object:

```json
{
  "timestamp": 1775700000
}
```

## Video Frame Payload

Binary layout:

1. `timestamp` - 8 bytes unsigned big-endian
2. `width` - 2 bytes unsigned big-endian
3. `height` - 2 bytes unsigned big-endian
4. `jpegByteCount` - 4 bytes unsigned big-endian
5. `jpegData` - JPEG bytes

Rules:

- Camera sends at approximately 10 fps maximum
- Supervisor replaces `latestFrame` on receipt
- Frames are memory-only and never written to disk

## Detection Results Payload

JSON object:

```json
{
  "timestamp": 1775700000,
  "results": [
    {
      "trackID": 7,
      "score": -0.82,
      "label": "normal",
      "boundingBox": { "x": 120, "y": 84, "width": 210, "height": 430 },
      "keypoints": [
        { "x": 140.0, "y": 105.0, "confidence": 0.91 }
      ]
    }
  ]
}
```

Rules:

- `label` values: `warmup`, `normal`, `anomaly`
- Keypoints use 18-point COCO18/OpenPose order
- Multiple tracked people are allowed in one message

## Disconnect Handling

- On explicit disconnect notice or 5-second timeout without heartbeat/frame traffic:
  - Session state becomes `stale` then `disconnected`
  - Supervisor keeps the last received frame visible
  - Supervisor overlays stale/disconnected status on the tile
