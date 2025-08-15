"""Telemetry module for metrics, logging, and tracing."""

import logging
import sys
from prometheus_client import Counter, Histogram, Gauge
from ..config import settings

# Setup structured logging
def setup_logging():
    """Setup structured logging."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("inzwa")

def get_logger(name: str):
    """Get a logger instance."""
    return logging.getLogger(name)

# Metrics
# ASR metrics
asr_latency = Histogram(
    "inzwa_asr_latency_ms",
    "ASR processing latency in milliseconds",
    buckets=(100, 200, 300, 500, 1000, 2000, 5000)
)
asr_partial_latency = Histogram(
    "inzwa_asr_partial_latency_ms",
    "ASR partial result latency"
)

# LLM metrics
llm_ttfw_histogram = Histogram(
    "inzwa_llm_ttfw_ms",
    "LLM time to first word in milliseconds",
    buckets=(100, 200, 500, 1000, 2000, 5000)
)
llm_tokens_per_second = Histogram(
    "inzwa_llm_tokens_per_second",
    "LLM token generation rate"
)

# TTS metrics
tts_rtf_histogram = Histogram(
    "inzwa_tts_rtf",
    "TTS real-time factor (1.0 = realtime)",
    buckets=(0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0)
)

# Session metrics
session_active = Gauge(
    "inzwa_session_active",
    "Number of active sessions"
)
session_duration = Histogram(
    "inzwa_session_duration_seconds",
    "Session duration in seconds",
    buckets=(10, 30, 60, 120, 300, 600)
)

# Error counters
error_counter = Counter(
    "inzwa_errors_total",
    "Total errors by component",
    ["component", "error_type"]
)

# Request counters
request_counter = Counter(
    "inzwa_requests_total",
    "Total API requests",
    ["endpoint", "method", "status"]
)

__all__ = [
    "setup_logging",
    "get_logger",
    "asr_latency",
    "asr_partial_latency",
    "llm_ttfw_histogram",
    "llm_tokens_per_second",
    "tts_rtf_histogram",
    "session_active",
    "session_duration",
    "error_counter",
    "request_counter"
]
