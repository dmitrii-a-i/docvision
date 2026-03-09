"""API smoke tests with mocked pipeline (no GPU/models needed)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.schemas import Detection, ProcessResponse


@pytest.fixture
def client():
    """Create a TestClient with a mocked pipeline."""
    dummy_response = ProcessResponse(
        fields={"name": "TEST", "date": "01.01.2000"},
        detections=[
            Detection(label="text", confidence=0.95, bbox=[10, 20, 300, 50]),
            Detection(label="photo", confidence=0.90, bbox=[50, 100, 250, 400]),
        ],
        dewarped_image="AAAA",
        annotated_image="BBBB",
    )

    mock_pipeline = MagicMock()
    mock_pipeline.process.return_value = dummy_response
    mock_pipeline.models_loaded = ["corner_detect", "field_detect", "vlm_test"]

    with patch("app.main.Pipeline", return_value=mock_pipeline):
        from app.main import app
        import app.main as main_mod
        main_mod.pipeline = mock_pipeline

        yield TestClient(app)

        main_mod.pipeline = None


class TestHealthEndpoint:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert len(data["models_loaded"]) == 3

    def test_health_loading(self):
        """Pipeline is None → status=loading."""
        with patch("app.main.Pipeline", return_value=None):
            from app.main import app
            import app.main as main_mod
            old = main_mod.pipeline
            main_mod.pipeline = None
            try:
                r = TestClient(app).get("/health")
                assert r.status_code == 200
                assert r.json()["status"] == "loading"
            finally:
                main_mod.pipeline = old


class TestIndexEndpoint:
    def test_returns_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "DocVision" in r.text


class TestProcessEndpoint:
    def test_empty_file(self, client):
        r = client.post("/process", files={"file": ("empty.jpg", b"", "image/jpeg")})
        assert r.status_code == 400

    def test_invalid_image(self, client):
        from unittest.mock import MagicMock
        import app.main as main_mod
        main_mod.pipeline.process.side_effect = ValueError("Could not decode image")

        r = client.post(
            "/process",
            files={"file": ("bad.jpg", b"not an image", "image/jpeg")},
        )
        assert r.status_code == 422

        main_mod.pipeline.process.side_effect = None

    def test_valid_image(self, client):
        """Mock pipeline returns dummy response for any valid image bytes."""
        import cv2
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)

        r = client.post(
            "/process",
            files={"file": ("test.jpg", buf.tobytes(), "image/jpeg")},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["fields"]["name"] == "TEST"
        assert len(data["detections"]) == 2
        assert data["dewarped_image"] == "AAAA"
        assert data["annotated_image"] == "BBBB"

    def test_no_file_field(self, client):
        r = client.post("/process")
        assert r.status_code == 422
