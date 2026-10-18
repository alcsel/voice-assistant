# Voice Assistant

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-39%20passing-brightgreen)

Record a question in the browser, turn the recording into text with
[Vosk](https://alphacephei.com/vosk/), optionally ask a
[Hugging Face](https://huggingface.co/) model for an answer and listen to it
through SpeechT5.

Everything runs locally: the only network calls are the optional Hugging Face
question answering and text to speech models.

## Features

- Browser recorder page with live status, playback, copy and download.
- Speech to text with Vosk, validated for mono 16-bit audio before the model
  is touched, so bad uploads fail with a readable message.
- Question answering against any Hugging Face text generation endpoint.
- Text to speech with the SpeechT5 model family, loaded lazily.
- Upload size and request timeouts configurable through the environment.

## Requirements

| Component | Notes |
| --- | --- |
| Python | 3.10 or newer |
| FFmpeg | must be on `PATH`, used to normalise uploads to 16 kHz mono WAV |
| Vosk model | download from <https://alphacephei.com/vosk/models> into `models/` |
| Hugging Face token | only needed for the `ask` command and the speech synthesis |

The default Vosk model is `vosk-model-small-tr-0.3` (Turkish). Point
`VOSK_MODEL` at the directory of any other model to change the language.

## Installation

```bash
git clone https://github.com/alcsel/voice-assistant.git
cd aisoundassistant
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download a Vosk model and place it under `models/`:

```bash
mkdir -p models
curl -LO https://alphacephei.com/vosk/models/vosk-model-small-tr-0.3.zip
unzip vosk-model-small-tr-0.3.zip -d models
```

## Usage

Start the browser recorder:

```bash
python -m voice_assistant.cli serve
```

Open <http://127.0.0.1:5000>, press **Start recording**, speak, then press the
button again. The transcript appears in the text box and can be copied or
downloaded.

Transcribe an existing WAV file:

```bash
python -m voice_assistant.cli transcribe recording.wav
python -m voice_assistant.cli transcribe recording.wav -o transcript.txt
```

Ask a question (needs `HF_TOKEN`):

```bash
export HF_TOKEN=hf_...
python -m voice_assistant.cli ask -q "What is a neural network?"
python -m voice_assistant.cli ask -f questions.txt -o answers.txt
```

Synthesise speech:

```bash
python -m voice_assistant.cli say "Hello there" -o hello.wav
```

## Configuration

All settings are read from the environment.

| Variable | Default | Purpose |
| --- | --- | --- |
| `HF_TOKEN` | _unset_ | Hugging Face token. Also read from `HUGGINGFACE_API_TOKEN` and `HUGGINGFACEHUB_API_TOKEN`. |
| `HF_MODEL` | `HuggingFaceH4/zephyr-7b-beta` | Text generation model for `ask`. |
| `VOSK_MODEL` | `models/vosk-model-small-tr-0.3` | Vosk model directory. |
| `FFMPEG_BINARY` | `ffmpeg` | Path to the FFmpeg executable. |
| `VOICE_ASSISTANT_MAX_UPLOAD_MB` | `25` | Largest accepted recording. |
| `VOICE_ASSISTANT_TIMEOUT` | `60` | HTTP timeout in seconds for Hugging Face calls. |

## HTTP API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | The recorder page. |
| `GET` | `/api/health` | Service status, model path and whether question answering is configured. |
| `POST` | `/api/transcribe` | Multipart upload in the `audio` field. Returns `text/plain` on success, JSON `{"error": ...}` with 400, 413 or 422 on failure. |

```bash
curl -F "audio=@recording.webm" http://127.0.0.1:5000/api/transcribe
```

## Project layout

```
voice_assistant/
  config.py       environment driven settings
  transcribe.py   WAV validation and Vosk transcription
  qa.py           Hugging Face question answering
  speech.py       SpeechT5 text to speech
  server.py       Flask application and upload endpoint
  cli.py          transcribe / ask / say / serve commands
static/index.html browser recorder
tests/            pytest suite, no network or model downloads required
```

## Tests

```bash
python -m pytest
```

The suite stubs FFmpeg and the speech models, so it runs offline and stays fast.

## License

[MIT](LICENSE) © 2026 Selim Calici
