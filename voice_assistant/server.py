"""Flask application that turns uploaded recordings into text."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

from .config import Settings
from .transcribe import TranscriptionError, load_model, transcribe_wave

__all__ = ["create_app"]

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def _convert_to_wav(source: Path, destination: Path, ffmpeg_binary: str) -> None:
    completed = subprocess.run(
        [
            ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-sample_fmt",
            "s16",
            str(destination),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise TranscriptionError(f"ffmpeg could not convert the recording: {detail}")


def create_app(settings: Settings | None = None, model=None) -> Flask:
    """Build the Flask app.

    ``model`` accepts a pre-built Vosk model so tests never touch the network
    or a multi-hundred-megabyte download.
    """
    resolved = settings or Settings.from_env()
    app = Flask(__name__, static_folder=str(_STATIC_DIR), static_url_path="")
    app.config["MAX_CONTENT_LENGTH"] = resolved.max_upload_bytes

    state: dict[str, object] = {"model": model}

    def get_model():
        if state["model"] is None:
            state["model"] = load_model(resolved.vosk_model_path)
        return state["model"]

    @app.get("/")
    def index():
        return send_from_directory(_STATIC_DIR, "index.html")

    @app.get("/api/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "transcription_model": str(resolved.vosk_model_path),
                "question_answering": resolved.is_qa_configured,
                "max_upload_mb": resolved.max_upload_mb,
            }
        )

    @app.post("/api/transcribe")
    def transcribe():
        upload = request.files.get("audio")
        if upload is None or not upload.filename:
            return jsonify({"error": "no audio file was uploaded"}), 400

        with tempfile.TemporaryDirectory(prefix="voice-assistant-") as workspace:
            folder = Path(workspace)
            uploaded = folder / "upload.bin"
            converted = folder / "audio.wav"
            upload.save(str(uploaded))

            try:
                _convert_to_wav(uploaded, converted, resolved.ffmpeg_binary)
                text = transcribe_wave(converted, get_model())
            except TranscriptionError as error:
                return jsonify({"error": str(error)}), 422

        if not text:
            return jsonify({"error": "no speech was recognised in the recording"}), 422

        response = Response(text, mimetype="text/plain; charset=utf-8")
        response.headers["Content-Disposition"] = 'attachment; filename="transcript.txt"'
        return response

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify({"error": f"recording exceeds the {resolved.max_upload_mb} MB limit"}), 413

    return app


def main() -> None:
    settings = Settings.from_env()
    create_app(settings).run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
