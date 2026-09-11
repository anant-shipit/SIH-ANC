"""
test_dashboard_api.py — Unit and integration tests for the GTCRN Model Dashboard API.
"""
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from sih26052.dashboard.server import create_app

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_WAV = REPO_ROOT / "data" / "test_audio" / "test_0000_noisy.wav"
CLEAN_WAV = REPO_ROOT / "data" / "test_audio" / "test_0000_clean.wav"


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_index_page(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "GTCRN SPEECH ENHANCEMENT" in res.text
    assert "app.js" in res.text


def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["model_ready"] is True
    assert "pytorch" in data["available_engines"]


def test_models_endpoint(client):
    res = client.get("/api/models")
    assert res.status_code == 200
    data = res.json()
    assert "engines" in data
    engine_ids = [e["id"] for e in data["engines"]]
    assert "pytorch" in engine_ids
    assert "onnx_stream_int8" in engine_ids


def test_presets_endpoint(client):
    res = client.get("/api/presets")
    assert res.status_code == 200
    data = res.json()
    assert "presets" in data
    assert len(data["presets"]) > 0
    preset = data["presets"][0]
    assert "id" in preset
    assert "filename" in preset


def test_preset_audio_endpoint(client):
    res = client.get("/api/presets")
    first_id = res.json()["presets"][0]["id"]

    res2 = client.get(f"/api/preset/{first_id}")
    assert res2.status_code == 200
    data = res2.json()
    assert data["sample_rate"] == 16000
    assert data["duration_sec"] > 0
    assert data["noisy_audio_url"].startswith("data:audio/wav;base64,")


def test_enhance_audio_pytorch(client):
    if not TEST_WAV.exists():
        pytest.skip("Test WAV file not found")

    with open(TEST_WAV, "rb") as f:
        files = {"file": ("test_0000_noisy.wav", f, "audio/wav")}
        data = {"engine": "pytorch"}
        res = client.post("/api/enhance", files=files, data=data)

    assert res.status_code == 200
    result = res.json()
    assert result["status"] == "success"
    assert result["engine"] == "pytorch"
    assert result["enhanced_audio_url"].startswith("data:audio/wav;base64,")
    assert result["processing_time_ms"] > 0
    assert result["rtf"] > 0
    assert "metrics" in result
    assert result["metrics"]["snr_improvement_db"] >= 0
    assert "spectrograms" in result
    assert len(result["spectrograms"]["raw"]) == 64
    assert len(result["spectrograms"]["enhanced"]) == 64


def test_enhance_audio_onnx_int8(client):
    if not TEST_WAV.exists():
        pytest.skip("Test WAV file not found")

    with open(TEST_WAV, "rb") as f:
        files = {"file": ("test_0000_noisy.wav", f, "audio/wav")}
        data = {"engine": "onnx_stream_int8"}
        res = client.post("/api/enhance", files=files, data=data)

    assert res.status_code == 200
    result = res.json()
    assert result["status"] == "success"
    assert result["engine"] == "onnx_stream_int8"
    assert result["enhanced_audio_url"].startswith("data:audio/wav;base64,")
    assert result["processing_time_ms"] > 0
