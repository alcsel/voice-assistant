from __future__ import annotations

from pathlib import Path

import pytest

from voice_assistant.qa import AnswerError, _build_prompt, _extract_answer, ask_question
from voice_assistant.config import Settings


class _FakeResponse:
    def __init__(self, status_code, payload, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


def _settings(token: str | None = "hf_secret") -> Settings:
    return Settings.from_env({"HF_TOKEN": token} if token else {}, base_dir=Path("/srv/app"))


class TestPromptBuilding:
    def test_wraps_the_question_in_the_model_template(self):
        prompt = _build_prompt("What is a neural network?")
        assert prompt.startswith("<|system|>")
        assert "<|user|>What is a neural network?<|assistant|>" in prompt


class TestAnswerExtraction:
    def test_keeps_only_the_generated_reply(self):
        payload = [{"generated_text": "<|system|>x<|user|>y<|assistant|>A neural network is a model."}]
        assert _extract_answer(payload) == "A neural network is a model."

    def test_accepts_a_bare_completion(self):
        assert _extract_answer([{"generated_text": "  Hello there.  "}]) == "Hello there."

    def test_rejects_an_empty_payload(self):
        with pytest.raises(AnswerError):
            _extract_answer([])

    def test_rejects_an_unexpected_shape(self):
        with pytest.raises(AnswerError):
            _extract_answer({"error": "not a list"})


class TestAskQuestion:
    def test_rejects_an_empty_question(self):
        with pytest.raises(ValueError):
            ask_question("  ", _settings())

    def test_requires_a_configured_token(self):
        with pytest.raises(AnswerError, match="HF_TOKEN"):
            ask_question("hello", _settings(token=None))

    def test_returns_the_answer_on_success(self):
        captured = {}

        def transport(url, headers, json, timeout):
            captured.update(url=url, headers=headers, payload=json, timeout=timeout)
            return _FakeResponse(200, [{"generated_text": "<|assistant|>Forty two."}])

        answer = ask_question("What is the answer?", _settings(), transport=transport)

        assert answer == "Forty two."
        assert captured["headers"]["Authorization"] == "Bearer hf_secret"
        assert captured["payload"]["inputs"].endswith("<|assistant|>")
        assert captured["timeout"] == 60.0

    def test_surfaces_the_status_code_of_a_failed_call(self):
        transport = lambda **kwargs: _FakeResponse(503, None, "model is loading")

        with pytest.raises(AnswerError, match="503"):
            ask_question("hello", _settings(), transport=transport)

    def test_forwards_the_token_budget(self):
        captured = {}

        def transport(**kwargs):
            captured.update(kwargs)
            return _FakeResponse(200, [{"generated_text": "ok"}])

        ask_question("hello", _settings(), max_new_tokens=42, transport=transport)
        assert captured["json"]["parameters"]["max_new_tokens"] == 42
