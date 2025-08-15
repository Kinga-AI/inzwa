"""Backpressure management for streaming pipeline."""

from ..config import settings
from ..telemetry import get_logger

logger = get_logger(__name__)


class BackpressureManager:
    """Manages backpressure in the streaming pipeline."""
    
    def __init__(self):
        self.threshold = settings.backpressure_threshold
        self.drop_count = 0
        self.total_count = 0
    
    def should_drop(self, queue_size: int) -> bool:
        """Determine if we should drop data due to backpressure."""
        self.total_count += 1
        
        if queue_size > self.threshold:
            self.drop_count += 1
            if self.drop_count % 10 == 0:
                logger.warning(
                    f"Backpressure: dropped {self.drop_count}/{self.total_count} items"
                )
            return True
        
        return False
    
    def get_stats(self) -> dict:
        """Get backpressure statistics."""
        return {
            "total": self.total_count,
            "dropped": self.drop_count,
            "drop_rate": self.drop_count / max(1, self.total_count)
        }
