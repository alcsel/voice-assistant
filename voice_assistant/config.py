"""Environment driven configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

__all__ = ["DEFAULT_HF_MODEL", "DEFAULT_VOSK_MODEL", "Settings"]

DEFAULT_HF_MODEL = "HuggingFaceH4/zephyr-7b-beta"
DEFAULT_VOSK_MODEL = "vosk-model-small-tr-0.3"
DEFAULT_MAX_UPLOAD_MB = 25
DEFAULT_TIMEOUT_SECONDS = 60.0


def _first_present(env: Mapping[str, str], *names: str) -> str | None:
    for name in names:
        value = env.get(name)
        if value and value.strip():
            return value.strip()
    return None


@dataclass(frozen=True)
class Settings:
    """Runtime settings resolved from environment variables."""

    hf_token: str | None = None
    hf_model: str = DEFAULT_HF_MODEL
    vosk_model_path: Path = Path("models") / DEFAULT_VOSK_MODEL
    max_upload_mb: int = DEFAULT_MAX_UPLOAD_MB
    request_timeout: float = DEFAULT_TIMEOUT_SECONDS
    ffmpeg_binary: str = "ffmpeg"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def is_qa_configured(self) -> bool:
        return bool(self.hf_token)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, base_dir: Path | None = None) -> "Settings":
        """Build settings from the process environment or a supplied mapping."""
        source = os.environ if env is None else env
        root = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent

        token = _first_present(source, "HF_TOKEN", "HUGGINGFACE_API_TOKEN", "HUGGINGFACEHUB_API_TOKEN")
        model = _first_present(source, "HF_MODEL", "VOICE_ASSISTANT_HF_MODEL") or DEFAULT_HF_MODEL

        vosk_name = _first_present(source, "VOSK_MODEL", "VOICE_ASSISTANT_VOSK_MODEL") or DEFAULT_VOSK_MODEL
        vosk_path = Path(vosk_name)
        if not vosk_path.is_absolute():
            vosk_path = root / "models" / vosk_path

        max_upload = _first_present(source, "VOICE_ASSISTANT_MAX_UPLOAD_MB")
        try:
            max_upload_mb = int(max_upload) if max_upload else DEFAULT_MAX_UPLOAD_MB
        except ValueError:
            max_upload_mb = DEFAULT_MAX_UPLOAD_MB

        timeout = _first_present(source, "VOICE_ASSISTANT_TIMEOUT")
        try:
            request_timeout = float(timeout) if timeout else DEFAULT_TIMEOUT_SECONDS
        except ValueError:
            request_timeout = DEFAULT_TIMEOUT_SECONDS

        return cls(
            hf_token=token,
            hf_model=model,
            vosk_model_path=vosk_path,
            max_upload_mb=max(1, max_upload_mb),
            request_timeout=max(1.0, request_timeout),
            ffmpeg_binary=_first_present(source, "FFMPEG_BINARY") or "ffmpeg",
        )
