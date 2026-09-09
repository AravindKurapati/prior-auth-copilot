from pathlib import Path

import pytest

from pa_copilot.config import Settings, load_settings


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    (tmp_path / "models.yaml").write_text(
        "agent: gemini-flash-latest\n"
        "summarizer: gemini-flash-lite-latest\n"
        "temperature_agent: 0.0\n"
    )
    (tmp_path / "routing.yaml").write_text(
        "max_replans: 2\nmax_hops: 12\nrecursion_limit: 40\ntau: 0.55\n"
    )
    (tmp_path / "memory.yaml").write_text(
        "namespaces:\n"
        "  episodic: {ttl_days: 90, cap: 50}\n"
        "  member: {ttl_days: 365, cap: 100}\n"
        "importance_weights: {routine: 1.0, notable: 1.5, critical: 3.0}\n"
    )
    return tmp_path


def test_loads_yaml_values(config_dir: Path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    s = load_settings(config_dir=config_dir, env_file=None)
    assert isinstance(s, Settings)
    assert s.model_agent == "gemini-flash-latest"
    assert s.model_summarizer == "gemini-flash-lite-latest"
    assert s.max_replans == 2
    assert s.tau == 0.55
    assert s.memory["importance_weights"]["critical"] == 3.0
    assert s.gemini_api_key is None


def test_env_overrides_and_key_read(config_dir: Path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
    monkeypatch.setenv("PA_MODEL_AGENT", "gemini-pro-latest")
    s = load_settings(config_dir=config_dir, env_file=None)
    assert s.gemini_api_key == "test-key-123"
    assert s.model_agent == "gemini-pro-latest"


def test_frozen(config_dir: Path):
    s = load_settings(config_dir=config_dir, env_file=None)
    with pytest.raises(Exception):
        s.model_agent = "x"
