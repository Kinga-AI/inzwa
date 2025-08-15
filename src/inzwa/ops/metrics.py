"""Ultra-light metrics per .cursorrules."""

from prometheus_client import Counter, Histogram, Gauge

# Core metrics only - no bloat
METRICS = {
    # Latency metrics (per .cursorrules budgets)
    "asr_latency_ms": Histogram(
        "inzwa_asr_latency_ms",
        "ASR processing latency",
        buckets=(100, 200, 300, 500, 1000)
    ),
    "llm_ttfb_ms": Histogram(
        "inzwa_llm_ttfb_ms", 
        "LLM time to first byte",
        buckets=(200, 400, 600, 900, 1200)
    ),
    "tts_ttfw_ms": Histogram(
        "inzwa_tts_ttfw_ms",
        "TTS time to first word", 
        buckets=(100, 200, 300, 500)
    ),
    "roundtrip_ms": Histogram(
        "inzwa_roundtrip_ms",
        "End-to-end latency",
        buckets=(500, 800, 1200, 2000, 5000)
    ),
    
    # Throughput metrics
    "token_rate": Histogram(
        "inzwa_token_rate",
        "Tokens per second",
        buckets=(10, 20, 30, 50, 100, 200)
    ),
    "rtf": Histogram(
        "inzwa_rtf",
        "Real-time factor",
        buckets=(0.1, 0.2, 0.5, 0.8, 1.0, 1.5)
    ),
    
    # Session metrics
    "active_sessions": Gauge(
        "inzwa_active_sessions",
        "Currently active sessions"
    ),
    
    # Error tracking
    "errors": Counter(
        "inzwa_errors_total",
        "Total errors by component",
        ["component", "error_type"]
    )
}
