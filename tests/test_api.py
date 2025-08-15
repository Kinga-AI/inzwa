from fastapi.testclient import TestClient
from inzwa.api.app import app

client = TestClient(app)

def test_health():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
