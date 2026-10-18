"""Command line entry points for transcription, answering and speech."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Settings
from .qa import AnswerError, answer_file, ask_question
from .speech import synthesize
from .transcribe import TranscriptionError, load_model, transcribe_wave

__all__ = ["build_parser", "main"]


def _add_transcribe(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("transcribe", help="turn a WAV recording into text")
    parser.add_argument("audio", type=Path, help="mono 16-bit PCM WAV file")
    parser.add_argument("-o", "--output", type=Path, help="write the text here instead of stdout")


def _add_ask(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("ask", help="answer a question through Hugging Face")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-q", "--question", help="a single question")
    group.add_argument("-f", "--questions-file", type=Path, help="one question per line")
    parser.add_argument("-o", "--output", type=Path, help="write answers to this file")


def _add_say(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("say", help="render text to speech")
    parser.add_argument("text", help="the text to speak")
    parser.add_argument("-o", "--output", type=Path, default=Path("speech.wav"))
    parser.add_argument("--speaker", type=int, default=0, help="index into the CMU Arctic voices")


def _add_serve(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("serve", help="run the browser recorder")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voice-assistant", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_transcribe(subparsers)
    _add_ask(subparsers)
    _add_say(subparsers)
    _add_serve(subparsers)
    return parser


def _run_transcribe(args: argparse.Namespace, settings: Settings) -> int:
    text = transcribe_wave(args.audio, load_model(settings.vosk_model_path))
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


def _run_ask(args: argparse.Namespace, settings: Settings) -> int:
    if args.question:
        answers = [ask_question(args.question, settings)]
    elif args.output:
        count = answer_file(str(args.questions_file), str(args.output), settings)
        print(f"wrote {count} answers to {args.output}")
        return 0
    else:
        lines = args.questions_file.read_text(encoding="utf-8").splitlines()
        answers = [ask_question(line, settings) for line in lines if line.strip()]

    print("\n\n".join(answers))
    return 0


def _run_say(args: argparse.Namespace, _settings: Settings) -> int:
    from .speech import SpeechSynthesizer

    print(f"wrote {synthesize(args.text, args.output, synthesizer=SpeechSynthesizer(args.speaker))}")
    return 0


def _run_serve(args: argparse.Namespace, settings: Settings) -> int:
    from .server import create_app

    create_app(settings).run(host=args.host, port=args.port, debug=args.debug)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()

    handlers = {
        "transcribe": _run_transcribe,
        "ask": _run_ask,
        "say": _run_say,
        "serve": _run_serve,
    }

    try:
        return handlers[args.command](args, settings)
    except (TranscriptionError, AnswerError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
