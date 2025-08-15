## API Reference (Per .cursorrules - Minimal Surface)

### Core Endpoints Only

- `GET /healthz` - Liveness check
  - Response: `{"status":"ok"}`

- `GET /readyz` - Models ready, caches warm
  - Response: `{"ready": true, "models": {...}}`

- `POST /v1/admin/warmup` - Load models
  - Response: `{"versions": {...}, "checksums": {...}}`

- `POST /v1/chat` - LLM streaming (SSE/WS)
  - Request with Pydantic validation (max 400 chars):
```json
{ "messages": [{"role": "user", "content": "Mhoro"}], "stream": true }
```
  - Response: `text/event-stream` with ORJSON:
```
data: {"token": "M"}
data: {"token": "h"}
```

- `POST /v1/tts` - Text to audio stream
  - Request with validation (max 400 chars):
```json
{ "text": "Mhoro", "voice": "shona_female_a", "format": "opus" }
```
  - Response: `audio/opus` stream (20-40ms chunks)

### WebSocket `/ws/audio` (Primary Transport)

**Bidirectional audio + JSON events - default per .cursorrules**

- Connect: Optional `Authorization: Bearer <token>`
- Client → Server:
  - Audio: PCM16 16kHz, 20-40ms frames (binary)
  - Control (ORJSON):
```json
{"type":"start","session_id":"uuid","codec":"pcm16","sample_rate":16000}
{"type":"end_turn"}
```

- Server → Client:
  - Audio: Opus/PCM16 frames (binary)
  - Events (ORJSON):
```json
{"type":"asr.partial","text":"...","start_ms":0,"end_ms":320,"conf":0.92}
{"type":"llm.partial","token":"..."}
{"type":"tts.start"}
{"type":"tts.end"}
```

### WebRTC `/rtc/session` (Optional - Flag Only)

- Only enabled if `enable_webrtc=true`
- SRTP/DTLS for secure media
- Data channel mirrors WebSocket events

### Error Handling

- Safe JSON errors (no stack traces):
```json
{ "error": { "code": "RATE_LIMIT", "message": "Quota exceeded" } }
```
- Request timeout: 5s (configurable)
- Max text: 400 chars
- Max audio: 20 seconds

### Security

- API key/JWT with per-key quotas
- Strict CORS (no wildcards)
- No raw audio/text in logs
- HMAC user_hash only
- Rate limiting via token buckets


