from __future__ import annotations

import json
import struct
import wave
from pathlib import Path

import pytest

from voice_assistant.transcribe import (
    TranscriptionError,
    load_model,
    transcribe_wave,
    validate_wave,
)


def _write_wave(path: Path, *, channels=1, sample_width=2, frame_rate=16000, frames=8000) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(frame_rate)
        handle.writeframes(struct.pack("<" + "h" * frames, *([0] * frames)))
    return path


class _StubRecognizer:
    def __init__(self, partials, final):
        self._partials = list(partials)
        self._final = final

    def AcceptWaveform(self, data: bytes) -> bool:
        return bool(self._partials)

    def Result(self) -> str:
        return json.dumps({"text": self._partials.pop(0) if self._partials else ""})

    def FinalResult(self) -> str:
        return json.dumps({"text": self._final})


class _StubModel:
    def __init__(self, partials=(), final=""):
        self.partials = list(partials)
        self.final = final
        self.seen_rate = None

    def __call__(self, sample_rate):
        self.seen_rate = sample_rate
        return _StubRecognizer(self.partials, self.final)


class TestValidateWave:
    def test_accepts_a_mono_16_bit_recording(self, tmp_path: Path):
        path = _write_wave(tmp_path / "ok.wav")
        assert validate_wave(path) == 16000

    def test_reports_a_missing_file(self, tmp_path: Path):
        with pytest.raises(TranscriptionError, match="not found"):
            validate_wave(tmp_path / "missing.wav")

    def test_rejects_stereo_audio(self, tmp_path: Path):
        path = _write_wave(tmp_path / "stereo.wav", channels=2)
        with pytest.raises(TranscriptionError, match="mono"):
            validate_wave(path)

    def test_rejects_8_bit_samples(self, tmp_path: Path):
        path = _write_wave(tmp_path / "eight.wav", sample_width=1)
        with pytest.raises(TranscriptionError, match="16-bit"):
            validate_wave(path)

    def test_rejects_an_unsupported_sample_rate(self, tmp_path: Path):
        path = _write_wave(tmp_path / "odd.wav", frame_rate=11025)
        with pytest.raises(TranscriptionError, match="sample rate"):
            validate_wave(path)

    def test_reports_a_file_that_is_not_wave_at_all(self, tmp_path: Path):
        broken = tmp_path / "broken.wav"
        broken.write_bytes(b"definitely not audio")
        with pytest.raises(TranscriptionError, match="WAV"):
            validate_wave(broken)


class TestTranscribeWave:
    def test_joins_partial_and_final_results(self, tmp_path: Path):
        path = _write_wave(tmp_path / "speech.wav")
        model = _StubModel(["hello there"], "general kenobi")
        assert transcribe_wave(path, model) == "hello there general kenobi"

    def test_forwards_the_sample_rate_to_the_model(self, tmp_path: Path):
        path = _write_wave(tmp_path / "speech.wav", frame_rate=8000)
        model = _StubModel([], "a")
        transcribe_wave(path, model)
        assert model.seen_rate == 8000

    def test_returns_an_empty_string_when_nothing_is_recognised(self, tmp_path: Path):
        path = _write_wave(tmp_path / "silence.wav")
        assert transcribe_wave(path, _StubModel([], "")) == ""

    def test_validates_before_loading_the_model(self, tmp_path: Path):
        path = _write_wave(tmp_path / "stereo.wav", channels=2)
        model = _StubModel([], "x")
        with pytest.raises(TranscriptionError):
            transcribe_wave(path, model)
        assert model.seen_rate is None


class TestLoadModel:
    def test_explains_how_to_obtain_a_missing_model(self, tmp_path: Path):
        with pytest.raises(TranscriptionError, match="alphacephei"):
            load_model(tmp_path / "absent-model")
