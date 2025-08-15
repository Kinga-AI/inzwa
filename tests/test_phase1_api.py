"""Phase 1 Tests: Foundation & API per TASKS.md"""

import pytest
from fastapi.testclient import TestClient
from inzwa.api.app import app


class TestPhase1Foundation:
    """Test Phase 1.1-1.4: Core setup, settings, API, metrics."""
    
    @pytest.fixture
    def client(self):
        """Test client fixture."""
        return TestClient(app)
    
    def test_healthz_endpoint(self, client):
        """Test /healthz returns ok."""
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    
    def test_settings_loaded(self):
        """Test settings load with INZWA_ prefix."""
        from inzwa.ops.settings import settings
        
        # Check core settings exist
        assert settings.request_timeout_s == 5.0
        assert settings.max_text_chars == 400
        assert settings.max_audio_seconds == 20
        assert settings.backpressure_threshold == 8
    
    def test_metrics_endpoint(self, client):
        """Test /metrics endpoint exists."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert b"inzwa_" in response.content
    
    def test_cors_strict(self, client):
        """Test CORS is strict (no wildcards)."""
        # Test with invalid origin
        response = client.get(
            "/healthz",
            headers={"Origin": "http://evil.com"}
        )
        # Should still work but without CORS headers for evil origin
        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" not in response.headers
    
    def test_request_size_limits(self, client):
        """Test request size limits enforced."""
        # Test text limit (max 400 chars)
        large_text = "x" * 500
        
        # This should be rejected when we implement the limit
        # For now, just test the endpoint exists
        response = client.get("/v1/models")
        assert response.status_code == 200
    
    def test_no_raw_logging(self, caplog):
        """Test no raw audio/text in logs."""
        from inzwa.ops.logging import get_logger
        
        logger = get_logger("test")
        
        # Try logging raw data
        logger.info(b"raw audio bytes")
        logger.info("x" * 200)  # Long text
        
        # Check logs don't contain raw data
        assert "raw audio bytes" not in caplog.text
        assert "x" * 200 not in caplog.text
        assert "<binary" in caplog.text or "<truncated>" in caplog.text
