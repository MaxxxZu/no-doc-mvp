import pytest
from pydantic import ValidationError
from pydantic_settings.sources import SettingsError

VALUES = {
    "NDM_HOST": "127.0.0.2",
    "NDM_PORT": "9123",
    "NDM_ZSC_MODEL": "test/classifier",
    "NDM_NER_MODEL": "test/ner",
    "NDM_ZSC_REVISION": "a" * 40,
    "NDM_NER_REVISION": "b" * 40,
    "NDM_ZSC_MULTI_LABEL": "false",
    "NDM_ZSC_MAX_LEN": "123",
    "NDM_HF_HOME": "/unused/test-cache",
    "NDM_OFFLINE": "true",
    "NDM_CPU_THREADS": "3",
    "NDM_CORS_ORIGINS": '["https://allowed.example"]',
}


def test_defaults(monkeypatch):
    from config import Settings

    monkeypatch.delenv("HF_HOME")
    settings = Settings()
    assert settings.model_dump() == {
        "host": "0.0.0.0", "port": 8080, "cors_origins": [],
        "zsc_model": "NbAiLab/nb-bert-base-mnli",
        "ner_model": "NbAiLab/nb-bert-base-ner", "zsc_multi_label": True,
        "zsc_revision": "171cd472d61d91ad6609530997e63ea98515efcb",
        "ner_revision": "98083f61535300a7fb6d8b72c26873d47e81ac93",
        "zsc_max_len": 2048, "ner_max_len": 2048, "body_max_bytes": 65536,
        "hf_home": None, "offline": False, "cpu_threads": 2,
    }


@pytest.mark.parametrize("case", [str.upper, str.lower, str.title])
def test_env_dotenv_equivalence(monkeypatch, tmp_path, case):
    from config import Settings

    monkeypatch.delenv("HF_HOME")
    env_file = tmp_path / ".env"
    env_file.write_text("\n".join(f"{case(k)}={v}" for k, v in VALUES.items()))
    dotenv = Settings()
    env_file.unlink()
    for key, value in VALUES.items():
        monkeypatch.setenv(case(key), value)
    process = Settings()
    assert process == dotenv
    assert process.offline is True
    assert process.cpu_threads == 3
    assert process.port == 9123
    assert process.host == "127.0.0.2"
    assert process.zsc_model == "test/classifier"
    assert process.ner_model == "test/ner"
    assert process.zsc_revision == "a" * 40
    assert process.ner_revision == "b" * 40
    assert process.zsc_multi_label is False
    assert process.zsc_max_len == 123
    assert process.hf_home == "/unused/test-cache"
    assert process.cors_origins == ["https://allowed.example"]


@pytest.mark.parametrize("source", ["env", "dotenv"])
@pytest.mark.parametrize("primary,legacy,field,value", [
    ("NDM_ZSC_MODEL", "ZSC_MODEL", "zsc_model", "primary/model"),
    ("NDM_NER_MODEL", "NER_MODEL", "ner_model", "primary/model"),
    ("NDM_ZSC_MULTI_LABEL", "ZSC_MULTI_LABEL", "zsc_multi_label", False),
    ("NDM_ZSC_MAX_LEN", "ZSC_MAX_LEN", "zsc_max_len", 37),
    ("NDM_HF_HOME", "HF_HOME", "hf_home", "/primary/cache"),
    ("PORT", "NDM_PORT", "port", 9001),
])
def test_alias_priority(monkeypatch, tmp_path, source, primary, legacy, field, value):
    from config import Settings

    monkeypatch.delenv("HF_HOME")
    legacy_value = {"port": "9002", "zsc_max_len": "99", "zsc_multi_label": "true"}
    data = {primary: str(value), legacy: legacy_value.get(field, "legacy/value")}
    if source == "env":
        for key, val in data.items():
            monkeypatch.setenv(key, val)
    else:
        (tmp_path / ".env").write_text("\n".join(f"{k}={v}" for k, v in data.items()))
    assert getattr(Settings(), field) == value
    # The legacy alias remains usable on its own.
    if source == "env":
        monkeypatch.delenv(primary)
    else:
        (tmp_path / ".env").write_text(f"{legacy}={data[legacy]}")
    assert str(getattr(Settings(), field)).lower() == data[legacy].lower()


@pytest.mark.parametrize("process_key,dotenv_key", [
    ("NDM_PORT", "PORT"), ("PORT", "NDM_PORT"), ("NDM_PORT", "NDM_PORT"),
    ("ZSC_MODEL", "NDM_ZSC_MODEL"), ("NDM_ZSC_MODEL", "ZSC_MODEL"),
])
def test_source_priority(monkeypatch, tmp_path, process_key, dotenv_key):
    from config import Settings

    is_port = "PORT" in process_key
    process_value = "9003" if is_port else "process/model"
    dotenv_value = "9004" if is_port else "dotenv/model"
    (tmp_path / ".env").write_text(f"{dotenv_key}={dotenv_value}")
    monkeypatch.setenv(process_key, process_value)
    settings = Settings()
    assert str(settings.port if is_port else settings.zsc_model) == process_value
    assert Settings(port=9005).port == 9005


@pytest.mark.parametrize("source", ["env", "dotenv"])
@pytest.mark.parametrize("key,value", [
    ("NDM_CPU_THREADS", "0"), ("NDM_CPU_THREADS", "257"),
    ("NDM_CPU_THREADS", "abc"), ("NDM_OFFLINE", "maybe"), ("NDM_HF_HOME", ""),
    ("NDM_PORT", "0"), ("PORT", "65536"), ("NDM_PORT", "abc"),
    ("NDM_ZSC_MAX_LEN", "0"), ("NDM_ZSC_MAX_LEN", "-1"),
    ("NDM_ZSC_REVISION", "main"), ("NDM_NER_REVISION", ""),
    ("NDM_NER_REVISION", "g" * 40),
    ("NDM_ZSC_MODEL", ""), ("NDM_NER_MODEL", "   "),
    ("NDM_ZSC_MULTI_LABEL", "maybe"),
    ("NDM_CORS_ORIGINS", "https://a.example,https://b.example"),
    ("NDM_CORS_ORIGINS", ""), ("NDM_CORS_ORIGINS", "[broken"), ("NDM_CORS_ORIGINS", '{}'),
    ("NDM_CORS_ORIGINS", '"https://a.example"'),
    ("NDM_CORS_ORIGINS", '[1]'), ("NDM_CORS_ORIGINS", 'null'),
])
def test_invalid_settings(monkeypatch, tmp_path, source, key, value):
    from config import Settings

    if key == "NDM_HF_HOME":
        monkeypatch.delenv("HF_HOME")
    if source == "env":
        monkeypatch.setenv(key, value)
    else:
        (tmp_path / ".env").write_text(f'{key}={value}')
    with pytest.raises((ValidationError, SettingsError)):
        Settings()


def test_explicit_empty_cors(monkeypatch):
    from config import Settings

    monkeypatch.setenv("NDM_CORS_ORIGINS", "[]")
    assert Settings().cors_origins == []


def test_dotenv_can_be_disabled(tmp_path):
    from config import Settings

    (tmp_path / ".env").write_text("PORT=invalid")
    assert Settings(_env_file=None).port == 8080


def test_unrelated_dotenv_keys_ignored(tmp_path):
    from config import Settings

    (tmp_path / ".env").write_text("HOST=ignored\nCORS_ORIGINS=ignored\nUNRELATED=1")
    assert Settings().host == "0.0.0.0"
    assert Settings().cors_origins == []


def test_invalid_config_aborts_module_import(monkeypatch):
    from importlib.util import module_from_spec, spec_from_file_location

    import config

    monkeypatch.setenv("PORT", "0")
    spec = spec_from_file_location("invalid_config", config.__file__)
    module = module_from_spec(spec)
    with pytest.raises(ValidationError):
        spec.loader.exec_module(module)


@pytest.mark.parametrize("port", [1, 65535])
def test_valid_boundaries(port):
    from config import Settings

    settings = Settings(port=port, zsc_max_len=1)
    assert settings.port == port
    assert settings.zsc_max_len == 1
