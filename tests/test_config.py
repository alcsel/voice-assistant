from __future__ import annotations

from pathlib import Path

from voice_assistant.config import DEFAULT_HF_MODEL, Settings


class TestSettingsFromEnv:
    def test_falls_back_to_documented_defaults(self):
        settings = Settings.from_env({}, base_dir=Path("/srv/app"))
        assert settings.hf_token is None
        assert settings.hf_model == DEFAULT_HF_MODEL
        assert settings.max_upload_mb == 25
        assert settings.request_timeout == 60.0

    def test_reads_the_token_from_any_supported_name(self):
        for name in ("HF_TOKEN", "HUGGINGFACE_API_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
            settings = Settings.from_env({name: "hf_secret"}, base_dir=Path("/srv/app"))
            assert settings.hf_token == "hf_secret"
            assert settings.is_qa_configured

    def test_strips_whitespace_from_values(self):
        settings = Settings.from_env({"HF_TOKEN": "  hf_secret  "}, base_dir=Path("/srv/app"))
        assert settings.hf_token == "hf_secret"

    def test_ignores_blank_values(self):
        settings = Settings.from_env({"HF_TOKEN": "   "}, base_dir=Path("/srv/app"))
        assert settings.hf_token is None

    def test_resolves_a_relative_vosk_model_below_the_project_root(self):
        settings = Settings.from_env({"VOSK_MODEL": "my-model"}, base_dir=Path("/srv/app"))
        assert settings.vosk_model_path == Path("/srv/app/models/my-model")

    def test_keeps_an_absolute_vosk_model_path(self):
        settings = Settings.from_env({"VOSK_MODEL": str(Path("/opt/models/tr"))}, base_dir=Path("/srv/app"))
        assert settings.vosk_model_path == Path("/opt/models/tr")

    def test_rejects_a_non_numeric_upload_limit(self):
        settings = Settings.from_env({"VOICE_ASSISTANT_MAX_UPLOAD_MB": "lots"}, base_dir=Path("/srv/app"))
        assert settings.max_upload_mb == 25

    def test_clamps_an_unusable_upload_limit(self):
        settings = Settings.from_env({"VOICE_ASSISTANT_MAX_UPLOAD_MB": "0"}, base_dir=Path("/srv/app"))
        assert settings.max_upload_mb == 1

    def test_converts_the_upload_limit_to_bytes(self):
        settings = Settings.from_env({"VOICE_ASSISTANT_MAX_UPLOAD_MB": "2"}, base_dir=Path("/srv/app"))
        assert settings.max_upload_bytes == 2 * 1024 * 1024

    def test_reads_a_custom_timeout(self):
        settings = Settings.from_env({"VOICE_ASSISTANT_TIMEOUT": "5.5"}, base_dir=Path("/srv/app"))
        assert settings.request_timeout == 5.5
