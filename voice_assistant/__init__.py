"""Voice assistant: speech capture, transcription and spoken answers."""

from __future__ import annotations

from .config import Settings
from .qa import ask_question
from .transcribe import transcribe_wave, validate_wave

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "Settings",
    "ask_question",
    "transcribe_wave",
    "validate_wave",
]
