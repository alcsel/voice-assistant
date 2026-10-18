"""Question answering through the Hugging Face inference API."""

from __future__ import annotations

from .config import Settings

__all__ = ["AnswerError", "answer_file", "ask_question"]

_ENDPOINT = "https://api-inference.huggingface.co/models/{model}"
_SYSTEM_PROMPT = "You are a concise assistant that answers in the user's language."


class AnswerError(RuntimeError):
    """Raised when the model endpoint cannot produce an answer."""


def _build_prompt(question: str) -> str:
    return f"<|system|>{_SYSTEM_PROMPT}<|user|>{question}<|assistant|>"


def _extract_answer(payload: object) -> str:
    if not isinstance(payload, list) or not payload:
        raise AnswerError(f"unexpected response payload: {payload!r}")

    first = payload[0]
    if not isinstance(first, dict) or "generated_text" not in first:
        raise AnswerError(f"unexpected response payload: {payload!r}")

    generated = str(first["generated_text"])
    if "<|assistant|>" in generated:
        generated = generated.rsplit("<|assistant|>", 1)[-1]

    return generated.strip()


def ask_question(
    question: str,
    settings: Settings,
    *,
    max_new_tokens: int = 200,
    transport=None,
) -> str:
    """Send a single question to the configured model and return its answer.

    ``transport`` is an injection point for tests; it receives the keyword
    arguments ``url``, ``headers``, ``json`` and ``timeout`` and must return an
    object with ``status_code``, ``text`` and ``json()`` attributes. By default
    it is :func:`requests.post`.
    """
    if not question.strip():
        raise ValueError("question must not be empty")
    if not settings.is_qa_configured:
        raise AnswerError(
            "no Hugging Face token configured; set HF_TOKEN in the environment "
            "before using the question answering feature"
        )

    if transport is None:
        import requests

        transport = requests.post

    response = transport(
        url=_ENDPOINT.format(model=settings.hf_model),
        headers={"Authorization": f"Bearer {settings.hf_token}"},
        json={"inputs": _build_prompt(question), "parameters": {"max_new_tokens": max_new_tokens}},
        timeout=settings.request_timeout,
    )

    if response.status_code != 200:
        raise AnswerError(f"model endpoint returned HTTP {response.status_code}: {response.text[:200]}")

    return _extract_answer(response.json())


def answer_file(questions_path: str, answers_path: str, settings: Settings) -> int:
    """Answer every non-empty line of a UTF-8 file and write the answers."""
    from pathlib import Path

    questions = [line.strip() for line in Path(questions_path).read_text(encoding="utf-8").splitlines()]
    questions = [line for line in questions if line]

    with Path(answers_path).open("w", encoding="utf-8") as handle:
        for question in questions:
            handle.write(f"{ask_question(question, settings)}\n\n")

    return len(questions)
