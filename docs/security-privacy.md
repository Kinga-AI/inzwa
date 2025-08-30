## Security and Privacy: Izwi

### Objectives

- Protect user audio/text and metadata end‑to‑end.
- Meet best practices for authentication, authorization, and secret handling.
- Minimize data retention; enable explicit user consent for data use.

### Threat Model (High‑Level)

- Network attackers intercepting audio/data → TLS 1.2+; WebRTC SRTP/DTLS for media
- Unauthorized API access → API keys/JWT, per‑key quotas, IP allow/deny lists
- Supply chain risks → pinned dependencies, SBOM, image scanning, signed artifacts
- Data exfiltration via logs → structured logs with redaction, no raw audio logging
- Model tampering → checksums, signed weights, version pinning, integrity checks on load

### Authentication and Authorization

- API keys (server‑generated) or JWT (issuer trusted) for Gateway endpoints
- Per‑key RBAC: scopes like `asr.read`, `chat.read`, `tts.read`, `admin`
- Rate limiting + quotas per key: sessions, tok/s, minutes/day

### Transport Security

- HTTPS/TLS for REST/WS
- WebRTC: SRTP for media, DTLS for key exchange; TLS between signaling endpoints

### Data Handling and Privacy

- Default: do not persist raw audio or transcripts
- Optional, explicit opt‑in: store short encrypted samples for QA with retention policies (e.g., 7–30 days)
- PII policy: redact names/identifiers if logged; avoid storing device identifiers
- Access controls: restrict artifact buckets and logs by role; audit access

### Logging and Telemetry

- Structured JSON logs; include session/turn IDs, exclude payloads
- Metrics: non‑PII numeric counters and timings
- Traces: spans with IDs; no content bodies

### Secrets Management

- Use environment secrets (Fly.io, GH Actions) or HashiCorp Vault
- Rotate regularly; never commit secrets to VCS

### Dependency and Supply Chain Security

- Pin versions; use `pip-tools` or Poetry lock
- SBOM (syft); image scanning (Trivy); signed containers (cosign)

### Model Security

- Verify checksums before loading
- Validate model configs (no arbitrary code execution from model cards)

### Client Security Considerations

- Microphone permissions scoped to active session
- Disable arbitrary file downloads; sanitize filenames for recordings

### Compliance Notes

- Document data processing purposes; adhere to applicable data laws
- Provide user controls for opt‑in/out and data deletion where applicable

### Incident Response

- Intake via `security@kinga.ai` (see `SECURITY.md`)
- 90‑day coordinated disclosure target; triage severity; patch and rotate secrets if needed


