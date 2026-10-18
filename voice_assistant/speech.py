"""Text to speech through the SpeechT5 model family."""

from __future__ import annotations

from pathlib import Path

__all__ = ["SAMPLE_RATE", "SpeechSynthesizer", "synthesize"]

SAMPLE_RATE = 16000
_MODEL_ID = "microsoft/speecht5_tts"
_VOCODER_ID = "microsoft/speecht5_hifigan"
_SPEAKER_DATASET = "Matthijs/cmu-arctic-xvectors"


class SpeechSynthesizer:
    """Lazily loaded SpeechT5 pipeline.

    The three model files total several hundred megabytes, so nothing is
    downloaded until the first call to :meth:`say`.
    """

    def __init__(self, speaker_index: int = 0, device: str | None = None) -> None:
        self.speaker_index = speaker_index
        self._device = device
        self._processor = None
        self._model = None
        self._vocoder = None
        self._speaker_embedding = None

    @property
    def device(self) -> str:
        if self._device is None:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def _load(self) -> None:
        if self._model is not None:
            return

        import torch
        from datasets import load_dataset
        from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

        self._processor = SpeechT5Processor.from_pretrained(_MODEL_ID)
        self._model = SpeechT5ForTextToSpeech.from_pretrained(_MODEL_ID).to(self.device)
        self._vocoder = SpeechT5HifiGan.from_pretrained(_VOCODER_ID).to(self.device)

        voices = load_dataset(_SPEAKER_DATASET, split="validation")
        self._speaker_embedding = torch.tensor(voices[self.speaker_index]["xvector"]).unsqueeze(0).to(self.device)

    def say(self, text: str) -> object:
        """Return a 1-D numpy waveform for ``text`` at :data:`SAMPLE_RATE`."""
        if not text.strip():
            raise ValueError("text must not be empty")

        self._load()

        inputs = self._processor(text=text, return_tensors="pt").to(self.device)
        waveform = self._model.generate_speech(
            inputs["input_ids"],
            self._speaker_embedding,
            vocoder=self._vocoder,
        )
        return waveform.cpu().numpy()


def synthesize(
    text: str,
    output_path: str | Path,
    *,
    synthesizer: SpeechSynthesizer | None = None,
) -> Path:
    """Render ``text`` to a 16 kHz WAV file and return the written path."""
    import soundfile

    engine = synthesizer or SpeechSynthesizer()
    waveform = engine.say(text)

    target = Path(output_path).expanduser()
    if target.suffix.lower() != ".wav":
        target = target.with_suffix(".wav")
    target.parent.mkdir(parents=True, exist_ok=True)

    soundfile.write(str(target), waveform, samplerate=SAMPLE_RATE)
    return target
