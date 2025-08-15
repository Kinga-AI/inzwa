"""Safety filters for LLM input/output."""

import re
from typing import List, Dict, Any
from ..config import settings
from ..telemetry import get_logger

logger = get_logger(__name__)


class SafetyFilter:
    """Safety filter for LLM generation."""
    
    def __init__(self):
        self.enabled = settings.enable_safety_filters
        self.profanity_patterns = self._load_profanity_patterns()
        self.sensitive_patterns = self._load_sensitive_patterns()
        self.refusal_template = "I cannot provide that information."
    
    def _load_profanity_patterns(self) -> List[re.Pattern]:
        """Load profanity patterns."""
        # Placeholder - should load from config/file
        patterns = [
            # Add Shona-specific profanity patterns
        ]
        return [re.compile(p, re.IGNORECASE) for p in patterns]
    
    def _load_sensitive_patterns(self) -> List[re.Pattern]:
        """Load sensitive content patterns."""
        patterns = [
            r"\b(password|secret|private key)\b",
            # Add more patterns
        ]
        return [re.compile(p, re.IGNORECASE) for p in patterns]
    
    def filter_input(self, text: str) -> str:
        """Filter input text before sending to LLM."""
        if not self.enabled:
            return text
        
        # Check for profanity
        for pattern in self.profanity_patterns:
            if pattern.search(text):
                logger.warning("Profanity detected in input")
                text = pattern.sub("[FILTERED]", text)
        
        # Check for sensitive content
        for pattern in self.sensitive_patterns:
            if pattern.search(text):
                logger.warning("Sensitive content detected in input")
                text = pattern.sub("[REDACTED]", text)
        
        return text
    
    def filter_output(self, text: str) -> str:
        """Filter output text from LLM."""
        if not self.enabled:
            return text
        
        # Check for sensitive content in output
        for pattern in self.sensitive_patterns:
            if pattern.search(text):
                logger.warning("Sensitive content in output, replacing with refusal")
                return self.refusal_template
        
        # Check for profanity in output
        for pattern in self.profanity_patterns:
            if pattern.search(text):
                logger.warning("Profanity in output, filtering")
                text = pattern.sub("[*]", text)
        
        return text
    
    def is_safe(self, text: str) -> bool:
        """Check if text is safe."""
        if not self.enabled:
            return True
        
        for pattern in self.profanity_patterns + self.sensitive_patterns:
            if pattern.search(text):
                return False
        
        return True
