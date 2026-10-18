from __future__ import annotations

import io
import json
import struct
import wave
from pathlib import Path

import pytest

from voice_assistant import server
from voice_assistant.config import Settings


def _upload() -> io.BytesIO:
    return io.BytesIO(b"fake-webm-data")


def _write_wave(path: Path, frames: int = 400) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(struct.pack("<" + "h" * frames, *([0] * frames)))


class _StubRecognizer:
    def __init__(self, text: str) -> None:
        self._text = text

    def AcceptWaveform(self, _data: bytes) -> bool:
        return False

    def Result(self) -> str:
        return json.dumps({"text": ""})

    def FinalResult(self) -> str:
        return json.dumps({"text": self._text})


class _StubModel:
    def __init__(self, text: str = "hello world") -> None:
        self.text = text
        self.calls = 0

    def __call__(self, _sample_rate: float) -> _StubRecognizer:
        self.calls += 1
        return _StubRecognizer(self.text)


@pytest.fixture
def fake_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for ffmpeg so the tests never shell out to a real binary."""

    def _convert(_source: Path, destination: Path, _binary: str) -> None:
        _write_wave(destination)

    monkeypatch.setattr(server, "_convert_to_wav", _convert)


@pytest.fixture
def client(fake_ffmpeg: None):
    app = server.create_app(Settings(max_upload_mb=1), model=_StubModel())
    app.config.update(TESTING=True)
    return app.test_client()


class TestHealth:
    def test_reports_ok_and_the_model_path(self, client):
        payload = client.get("/api/health").get_json()
        assert payload["status"] == "ok"
        assert payload["max_upload_mb"] == 1

    def test_reports_question_answering_as_disabled_without_a_token(self, client):
        assert client.get("/api/health").get_json()["question_answering"] is False


class TestTranscribeEndpoint:
    def test_returns_a_text_attachment(self, client):
        response = client.post(
            "/api/transcribe",
            data={"audio": (_upload(), "recording.webm")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 200
        assert response.get_data(as_text=True) == "hello world"
        assert "transcript.txt" in response.headers["Content-Disposition"]

    def test_rejects_a_request_without_a_file(self, client):
        response = client.post("/api/transcribe", data={}, content_type="multipart/form-data")
        assert response.status_code == 400
        assert "error" in response.get_json()

    def test_reports_unreadable_audio_as_unprocessable(self, client, monkeypatch: pytest.MonkeyPatch):
        def _fail(_source, _destination, _binary):
            from voice_assistant.transcribe import TranscriptionError

            raise TranscriptionError("ffmpeg could not convert the recording: bad input")

        monkeypatch.setattr(server, "_convert_to_wav", _fail)
        response = client.post(
            "/api/transcribe",
            data={"audio": (_upload(), "recording.webm")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 422
        assert "bad input" in response.get_json()["error"]

    def test_reports_silence_as_unprocessable(self, client):
        app = server.create_app(Settings(max_upload_mb=1), model=_StubModel(""))
        app.config.update(TESTING=True)
        response = app.test_client().post(
            "/api/transcribe",
            data={"audio": (_upload(), "recording.webm")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 422
        assert "no speech" in response.get_json()["error"]


class TestUploadLimit:
    def test_rejects_a_recording_over_the_configured_limit(self, client):
        oversized = io.BytesIO(b"x" * (1024 * 1024 + 64))
        response = client.post(
            "/api/transcribe",
            data={"audio": (oversized, "huge.webm")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 413
        assert "1 MB" in response.get_json()["error"]


class TestIndex:
    def test_serves_the_recorder_page(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["Content-Type"]
