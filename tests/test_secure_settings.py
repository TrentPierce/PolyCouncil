import json

import council
from core import settings_store


def test_save_settings_strips_api_key_and_load_restores_from_secure_store(tmp_path, monkeypatch):
    settings_path = tmp_path / "settings.json"
    secrets = {}

    monkeypatch.setattr(settings_store, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(settings_store, "set_secret", lambda name, value: secrets.__setitem__(name, value) or True)
    monkeypatch.setattr(settings_store, "get_secret", lambda name: secrets.get(name, ""))
    monkeypatch.setattr(settings_store, "secure_store_available", lambda: True)

    ok = council.save_settings(
        {
            "provider_type": council.PROVIDER_OPENAI_COMPAT,
            "api_key": "sk-test",
            "provider_profiles": [
                {"id": "profile-1", "provider_type": council.PROVIDER_OPENAI_COMPAT, "api_key": "sk-profile"}
            ],
        }
    )

    assert ok is True
    raw = json.loads(settings_path.read_text(encoding="utf-8"))
    assert "api_key" not in raw
    assert "api_key" not in raw["provider_profiles"][0]
    assert secrets[settings_store.ACTIVE_API_KEY_REF] == "sk-test"
    assert secrets[settings_store._profile_secret_name("profile-1")] == "sk-profile"

    loaded = council.load_settings()
    assert loaded["api_key"] == "sk-test"
    assert loaded["provider_profiles"][0]["api_key"] == "sk-profile"


def test_save_settings_returns_false_when_secure_store_missing(tmp_path, monkeypatch):
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(settings_store, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(settings_store, "set_secret", None)
    monkeypatch.setattr(settings_store, "get_secret", None)
    monkeypatch.setattr(settings_store, "secure_store_available", lambda: False)

    ok = council.save_settings({"api_key": "sk-test"})

    assert ok is False
    raw = json.loads(settings_path.read_text(encoding="utf-8"))
    assert "api_key" not in raw


def test_load_settings_migrates_and_validates_schema(tmp_path, monkeypatch):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "provider_type": council.PROVIDER_LM_STUDIO,
                "max_concurrency": "99",
                "timeout_seconds": "9",
                "reduce_motion": "true",
                "persona_assignments": ["bad"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings_store, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(settings_store, "get_secret", lambda name: "")
    monkeypatch.setattr(settings_store, "secure_store_available", lambda: False)

    loaded = council.load_settings()

    assert loaded["schema_version"] == settings_store.SETTINGS_SCHEMA_VERSION
    assert loaded["settings_migrated_from_schema"] == 1
    assert loaded["max_concurrency"] == 8
    assert loaded["timeout_seconds"] == 15
    assert loaded["reduce_motion"] is True
    assert loaded["persona_assignments"] == {}
