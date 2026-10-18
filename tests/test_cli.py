from __future__ import annotations

from pathlib import Path

import pytest

from voice_assistant import cli


class TestParser:
    def test_requires_a_command(self):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args([])

    def test_rejects_missing_audio_arguments(self):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["transcribe"])

    def test_ask_requires_exactly_one_input(self):
        parser = cli.build_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(["ask"])
        with pytest.raises(SystemExit):
            parser.parse_args(["ask", "-q", "hi", "-f", "questions.txt"])

        assert parser.parse_args(["ask", "-q", "hi"]).question == "hi"
        assert parser.parse_args(["ask", "-f", "questions.txt"]).questions_file == Path("questions.txt")

    def test_serve_defaults_to_loopback(self):
        args = cli.build_parser().parse_args(["serve"])
        assert (args.host, args.port, args.debug) == ("127.0.0.1", 5000, False)


class TestCommands:
    def test_transcribe_writes_the_transcript(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
        monkeypatch.setattr(cli, "load_model", lambda _path: object())
        monkeypatch.setattr(cli, "transcribe_wave", lambda _path, _model: "hello world")

        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"")
        target = tmp_path / "transcript.txt"

        code = cli.main(["transcribe", str(audio), "-o", str(target)])

        assert code == 0
        assert target.read_text(encoding="utf-8") == "hello world\n"
        assert "wrote" in capsys.readouterr().out

    def test_reports_transcription_failures_without_a_traceback(
        self, monkeypatch: pytest.MonkeyPatch, capsys, tmp_path: Path
    ):
        from voice_assistant.transcribe import TranscriptionError

        def _fail(_path):
            raise TranscriptionError("Vosk model directory not found")

        monkeypatch.setattr(cli, "load_model", _fail)
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"")

        assert cli.main(["transcribe", str(audio)]) == 1
        assert "error: Vosk model directory not found" in capsys.readouterr().err

    def test_ask_answers_a_single_question(self, monkeypatch: pytest.MonkeyPatch, capsys):
        import voice_assistant.cli as module

        monkeypatch.setattr(module, "ask_question", lambda question, _settings: f"answer to {question}")

        assert cli.main(["ask", "-q", "why?"]) == 0
        assert capsys.readouterr().out.strip() == "answer to why?"
