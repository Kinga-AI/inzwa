# Inzwa MVP Development Tasks - Detailed Roadmap

## Overview
Build a state-of-the-art Shona voice assistant with <1s latency, following .cursorrules strictly.

**Success Metrics:**
- P50 TTFW ≤ 500ms
- P95 E2E ≤ 1200ms  
- Zero security vulnerabilities
- 100% .cursorrules compliance

**See detailed tasks in:**
- [Phase 1-2: Foundation & ASR](docs/tasks/phase1-2.md)
- [Phase 3-4: LLM & TTS](docs/tasks/phase3-4.md)
- [Phase 5-6: Orchestration & WebSocket](docs/tasks/phase5-6.md)
- [Phase 7-8: UI & Testing](docs/tasks/phase7-8.md)

---

## Quick Reference: Phase Overview

### Phase 1: Foundation & API Gateway (Week 1)
- 1.1 Project Setup & Configuration
- 1.2 FastAPI Base Application
- 1.3 Health & Readiness Endpoints
- 1.4 Authentication & Rate Limiting

### Phase 2: ASR Service (Week 2)
- 2.1 ASR Engine Integration
- 2.2 Voice Activity Detection (VAD)
- 2.3 WebSocket ASR Endpoint

### Phase 3: LLM Service (Week 3)
- 3.1 LLM Engine Integration
- 3.2 Safety Filters

### Phase 4: TTS Service (Week 4)
- 4.1 TTS Engine Integration

### Phase 5: Orchestrator (Week 5)
- 5.1 Pipeline Orchestration

### Phase 6: WebSocket Integration (Week 6)
- 6.1 Full Duplex WebSocket

### Phase 7: UI & Deployment (Week 7)
- 7.1 Minimal Web UI
- 7.2 Docker & Deployment

### Phase 8: Testing & Optimization (Week 8)
- 8.1 Performance Testing
- 8.2 Security Audit

---

## Definition of Done

Each task is considered DONE when:

1. **Code Complete**: Implementation matches spec
2. **Tests Pass**: All unit/integration tests green
3. **Performance Met**: Meets latency/resource budgets
4. **Security Verified**: No vulnerabilities or leaks
5. **Documentation Updated**: API docs and comments current
6. **Code Review Passed**: Approved by tech lead
7. **Metrics Tracked**: Performance metrics recorded
8. **Deployed**: Running in dev environment

---

## Testing Commands

```bash
# Run all tests
make test

# Run specific phase
./run_tests.sh 1

# Performance test
poetry run pytest tests/test_performance.py -v

# Security audit
make security-check

# Latency benchmark
poetry run python scripts/benchmark_latency.py

# Load test
poetry run locust -f tests/load_test.py --host http://localhost:8000

# Check .cursorrules compliance
python3 check_alignment.py
```

---

## Risk Register

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Shona data scarcity | High | High | Self-record, synthetic generation |
| Latency > 1s | High | Medium | Quantization, caching, GPU |
| Memory exhaustion | High | Low | Bounded queues, backpressure |
| Model quality poor | Medium | Medium | Iterative fine-tuning |
| Security breach | High | Low | Strict validation, sanitization |

---

## Daily Standup Template

```markdown
## Date: YYYY-MM-DD

### Yesterday
- Completed: [Task IDs]
- Blockers: [Issues]

### Today
- Working on: [Task IDs]
- Target: [Specific deliverables]

### Metrics
- P50 TTFW: XXXms
- P95 E2E: XXXms
- Tests: XX/XX passing
- Coverage: XX%

### Blockers
- [Issue]: [Mitigation plan]
```

---

**Remember**: Every line of code should be ultra-light, ultra-fast, and ultra-secure per .cursorrules!