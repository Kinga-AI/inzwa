"""Ultra-light logging per .cursorrules - no raw payloads."""

import logging
import sys
from typing import Any


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup minimal structured logging."""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger("inzwa")


def get_logger(name: str) -> logging.Logger:
    """Get logger - no raw audio/text logging."""
    logger = logging.getLogger(name)
    
    # Wrap to prevent raw payload logging
    original_log = logger._log
    
    def safe_log(level: int, msg: Any, args: Any, **kwargs: Any) -> None:
        # Strip any raw audio/text data
        if isinstance(msg, bytes):
            msg = f"<binary {len(msg)} bytes>"
        elif isinstance(msg, str) and len(msg) > 100:
            msg = f"{msg[:50]}...<truncated>"
        original_log(level, msg, args, **kwargs)
    
    logger._log = safe_log
    return logger
