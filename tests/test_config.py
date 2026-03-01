from config import _as_bool, get_settings


def test_as_bool_parses_common_truthy_values():
    assert _as_bool("1", False)
    assert _as_bool("true", False)
    assert _as_bool("Yes", False)


def test_get_settings_uses_defaults_when_optional_env_absent(monkeypatch):
    monkeypatch.delenv("SERVER_URL", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("LOG_TO_STDOUT", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.server_url == "http://localhost:8000"
    assert settings.log_level == "INFO"
