"""Dialogue state management."""

from typing import List, Dict, Any, Optional
from datetime import datetime


class DialogueState:
    """Manages conversation state and history."""
    
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.turns: List[Dict[str, Any]] = []
        self.session_metadata = {
            "start_time": datetime.utcnow(),
            "locale": "sn",  # Shona
            "voice_id": "shona_female_a"
        }
        self.safety_flags = {
            "profanity_detected": False,
            "sensitive_content": False
        }
    
    def add_user_turn(self, text: str, metadata: Optional[Dict] = None):
        """Add a user turn to the dialogue."""
        turn = {
            "role": "user",
            "content": text,
            "timestamp": datetime.utcnow(),
            "metadata": metadata or {}
        }
        self.turns.append(turn)
        self._trim_history()
    
    def add_assistant_turn(self, text: str, metadata: Optional[Dict] = None):
        """Add an assistant turn to the dialogue."""
        turn = {
            "role": "assistant",
            "content": text,
            "timestamp": datetime.utcnow(),
            "metadata": metadata or {}
        }
        self.turns.append(turn)
        self._trim_history()
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get messages formatted for LLM input."""
        messages = [
            {
                "role": "system",
                "content": self._get_system_prompt()
            }
        ]
        
        for turn in self.turns:
            messages.append({
                "role": turn["role"],
                "content": turn["content"]
            })
        
        return messages
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for Shona assistant."""
        return """You are a helpful Shona-speaking assistant. Respond naturally in Shona, 
        being culturally aware and using appropriate politeness markers. 
        Keep responses concise for natural conversation flow."""
    
    def _trim_history(self):
        """Trim conversation history to max turns."""
        if len(self.turns) > self.max_turns * 2:  # User + assistant turns
            self.turns = self.turns[-(self.max_turns * 2):]
    
    def clear(self):
        """Clear the dialogue state."""
        self.turns.clear()
        self.safety_flags = {
            "profanity_detected": False,
            "sensitive_content": False
        }
    
    def get_context_summary(self) -> str:
        """Get a summary of the conversation context."""
        if not self.turns:
            return "New conversation"
        
        # Simple summary: last few exchanges
        recent = self.turns[-4:] if len(self.turns) >= 4 else self.turns
        summary = " | ".join([f"{t['role']}: {t['content'][:50]}..." for t in recent])
        return summary
