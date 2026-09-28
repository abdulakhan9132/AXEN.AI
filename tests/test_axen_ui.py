"""
tests/test_axen_ui.py
=====================
Tests for the modern AXEN AI Web Interface and API endpoints.

Validates:
  1. Root HTML template serving and visual structure matching reference design
  2. Static assets (CSS, JS, Markdown parser)
  3. System status endpoint
  4. Conversation session creation, listing, switching, renaming, deleting
  5. Chat endpoint integration with AssistantController
  6. Deterministic tool verification through UI API
"""

import pytest

try:
    from fastapi.testclient import TestClient
    from app.ui.app import app
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    TestClient = None
    app = None


@pytest.fixture(scope="module")
def client():
    if not HAS_FASTAPI:
        pytest.skip("FastAPI not installed in current Python environment")
    with TestClient(app) as c:
        yield c


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI required for UI test suite")
class TestAxenUI:
    def test_serve_index_html(self, client):
        """Index page must serve HTML containing reference elements."""
        res = client.get("/")
        assert res.status_code == 200
        text = res.text
        # Visual structure verification
        assert "AXEN®" in text
        assert "NEW CHAT" in text
        assert "How Can I Help You Today?" in text
        assert "Ask Anything, Get Ideas, Write, Analyze, Design, Or Solve Problems With AI." in text
        assert "Message AI..." in text
        assert "btn-composer-send" in text
        assert "btn-composer-voice" in text

    def test_static_assets(self, client):
        """CSS and JS files must be reachable and contain core styling."""
        res_css = client.get("/static/css/styles.css")
        assert res_css.status_code == 200
        assert "--axen-orange" in res_css.text

        res_js = client.get("/static/js/app.js")
        assert res_js.status_code == 200
        assert "AXEN Frontend Controller" in res_js.text

        res_md = client.get("/static/js/markdown.js")
        assert res_md.status_code == 200
        assert "AxenMarkdown" in res_md.text

    def test_api_status(self, client):
        """Status endpoint returns model information and tools."""
        res = client.get("/api/status")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ready"
        assert "tools" in data
        assert "model_name" in data

    def test_session_lifecycle(self, client):
        """Verify session creation, listing, switching, renaming, and deletion."""
        # 1. Create session
        res = client.post("/api/sessions/new", json={"title": "Test Chat Session"})
        assert res.status_code == 200
        sid = res.json()["session_id"]
        assert sid

        # 2. List sessions
        res = client.get("/api/sessions")
        assert res.status_code == 200
        sessions = res.json()["sessions"]
        assert any(s["id"] == sid for s in sessions)

        # 3. Rename session
        res = client.patch(f"/api/sessions/{sid}", json={"title": "Updated Title"})
        assert res.status_code == 200
        assert res.json()["title"] == "Updated Title"

        # 4. Switch session
        res = client.post(f"/api/sessions/{sid}/switch")
        assert res.status_code == 200

        # 5. Delete session
        res = client.delete(f"/api/sessions/{sid}")
        assert res.status_code == 200

    def test_chat_calculation_tool(self, client):
        """Verify calculator tool is executed and verified through UI chat endpoint."""
        res = client.post("/api/chat", json={"message": "what is 25 * 40?"})
        assert res.status_code == 200
        data = res.json()
        assert "1000" in data["response"]
        assert data["tool_used"] is True
        assert data["tool_name"] == "calculator"
        assert data["status"] == "completed"

    def test_chat_legacy_goal_compatibility(self, client):
        """Ensure backward compatibility for payloads with 'goal' instead of 'message'."""
        res = client.post("/api/chat", json={"goal": "what is 10 + 15?"})
        assert res.status_code == 200
        data = res.json()
        assert "25" in data["response"]
