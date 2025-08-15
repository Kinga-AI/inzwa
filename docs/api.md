## API Reference

### REST Endpoints

- `GET /healthz`
  - 200 OK `{status:"ok"}`

- `GET /v1/models`
  - Response: list of ASR/LLM/TTS models, versions, sizes

- `POST /v1/asr`
  - Request (application/json):
```json
{ "audio_base64": "...", "sample_rate": 16000 }
```
  - Response:
```json
{ "text": "Mhoro", "segments": [{"text":"Mhoro","start_ms":0,"end_ms":320,"conf":0.92}] }
```

- `POST /v1/chat`
  - Request:
```json
{ "messages": [ {"role": "system", "content": "You are a helpful Shona assistant."}, {"role": "user", "content": "Mhoro"} ], "stream": true }
```
  - Response: `text/event-stream` with `data: {"token":"..."}` lines

- `POST /v1/tts`
  - Request:
```json
{ "text": "Mhoro", "voice": "shona_female_a", "format": "opus" }
```
  - Response: `audio/opus` stream

### WebSocket `/ws/audio`

- Connect with header `Authorization: Bearer <token>`
- Client → Server:
  - Control (JSON):
```json
{"type":"start","session_id":"uuid","codec":"pcm16","sample_rate":16000}
```
  - Audio (binary): PCM16 20–40 ms frames
  - Control (JSON): `{"type":"end_turn"}`

- Server → Client:
  - Events (JSON):
```json
{"type":"asr.partial","text":"Mhoro","start_ms":0,"end_ms":320,"conf":0.92}
{"type":"llm.partial","token":"Mh"}
{"type":"tts.start"}
{"type":"tts.end"}
```
  - Audio (binary): Opus/PCM16 frames

### WebRTC `/rtc/session`

- SDP Offer (POST) → Answer (JSON); then SRTP media; data channel mirrors WS events

### Errors

- Standard JSON error body:
```json
{ "error": { "code": "RATE_LIMIT", "message": "Quota exceeded" } }
```

### Auth

- API keys or JWT; pass via `Authorization: Bearer ...`


