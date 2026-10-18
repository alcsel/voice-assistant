"""Speech to text through Vosk."""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any, Protocol

__all__ = ["SUPPORTED_SAMPLE_RATES", "TranscriptionError", "transcribe_wave", "validate_wave"]

SUPPORTED_SAMPLE_RATES = (8000, 16000, 32000, 44100, 48000)
_CHUNK_FRAMES = 4000


class TranscriptionError(RuntimeError):
    """Raised when an audio file cannot be transcribed."""


class _Recognizer(Protocol):
    def AcceptWaveform(self, data: bytes) -> bool: ...
    def Result(self) -> str: ...
    def FinalResult(self) -> str: ...


class _Model(Protocol):
    def __call__(self, sample_rate: float) -> _Recognizer: ...


def validate_wave(path: str | Path) -> int:
    """Return the frame rate of a mono, 16-bit PCM WAV file.

    Vosk only accepts single channel 16-bit audio, so rejecting anything else
    up front produces a far clearer error than a crash deep inside the model.
    """
    target = Path(path)
    if not target.is_file():
        raise TranscriptionError(f"audio file not found: {target}")

    try:
        with wave.open(str(target), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            frame_rate = handle.getframerate()
    except wave.Error as error:
        raise TranscriptionError(f"not a readable WAV file: {error}") from error

    if channels != 1:
        raise TranscriptionError(f"expected mono audio, found {channels} channels")
    if sample_width != 2:
        raise TranscriptionError(f"expected 16-bit samples, found {sample_width * 8}-bit")
    if frame_rate not in SUPPORTED_SAMPLE_RATES:
        supported = ", ".join(str(rate) for rate in SUPPORTED_SAMPLE_RATES)
        raise TranscriptionError(f"unsupported sample rate {frame_rate} Hz, expected one of: {supported}")

    return frame_rate


def _collect(recognizer: _Recognizer, handle: Any) -> str:
    parts: list[str] = []

    while True:
        data = handle.readframes(_CHUNK_FRAMES)
        if not data:
            break
        if recognizer.AcceptWaveform(data):
            chunk = json.loads(recognizer.Result())
            text = chunk.get("text", "").strip()
            if text:
                parts.append(text)

    final = json.loads(recognizer.FinalResult())
    tail = final.get("text", "").strip()
    if tail:
        parts.append(tail)

    return " ".join(parts).strip()


def transcribe_wave(path: str | Path, model: _Model) -> str:
    """Transcribe a validated WAV file and return the recognised text."""
    frame_rate = validate_wave(path)

    recognizer = model(frame_rate)
    with wave.open(str(path), "rb") as handle:
        return _collect(recognizer, handle)


def load_model(model_path: str | Path) -> _Model:
    """Load a Vosk model from disk, with a helpful error when it is missing."""
    target = Path(model_path)
    if not target.is_dir():
        raise TranscriptionError(
            f"Vosk model directory not found: {target}. Download a model from "
            "https://alphacephei.com/vosk/models and point VOSK_MODEL at it."
        )

    from vosk import Model

    return Model(str(target))
