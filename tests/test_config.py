import os
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
    assert s.memory.importance_weights["critical"] == 3.0
    assert s.memory.namespaces["episodic"].ttl_days == 90
    assert s.memory.namespaces["episodic"].cap == 50
    assert s.gemini_api_key is None


def test_memory_config_is_frozen(config_dir: Path):
    s = load_settings(config_dir=config_dir, env_file=None)
    with pytest.raises(Exception):
        s.memory.recency_half_life_days = 999
    with pytest.raises(Exception):
        s.memory.namespaces["episodic"].cap = 999


def test_filesystem_anchors_resolve_to_real_dirs(config_dir: Path):
    s = load_settings(config_dir=config_dir, env_file=None)
    assert Path(s.repo_root).is_dir()
    assert Path(s.data_dir).is_dir()
    assert Path(s.synthetic_dir).is_dir()
    assert Path(s.samples_dir).is_dir()
    assert Path(s.traces_dir).is_dir()
    assert Path(s.synthetic_dir).parent == Path(s.data_dir)


def test_anchor_env_overrides(config_dir: Path, monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PA_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("PA_TEMPERATURE_AGENT", "0.7")
    s = load_settings(config_dir=config_dir, env_file=None)
    assert s.repo_root == str(tmp_path)
    assert s.data_dir == str(tmp_path / "data")
    assert s.temperature_agent == 0.7


def test_env_overrides_and_key_read(config_dir: Path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
    monkeypatch.setenv("PA_MODEL_AGENT", "gemini-pro-latest")
    s = load_settings(config_dir=config_dir, env_file=None)
    assert s.gemini_api_key == "test-key-123"
    assert s.model_agent == "gemini-pro-latest"


def test_gemini_api_key_mirrors_to_google_api_key(config_dir: Path, monkeypatch):
    """langchain-google-genai reads only GOOGLE_API_KEY; load_settings must mirror
    GEMINI_API_KEY onto it so the @slow AC-10 test can actually authenticate."""
    monkeypatch.setenv("GEMINI_API_KEY", "mirror-me-0123456789abcdef")
    # _env_snapshot (conftest) reverts the raw mirror write after this test
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    load_settings(config_dir=config_dir, env_file=None)
    assert os.environ["GOOGLE_API_KEY"] == "mirror-me-0123456789abcdef"


def test_existing_google_api_key_is_not_overwritten(config_dir: Path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-value")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-value-kept")
    load_settings(config_dir=config_dir, env_file=None)
    assert os.environ["GOOGLE_API_KEY"] == "google-value-kept"


def test_frozen(config_dir: Path):
    s = load_settings(config_dir=config_dir, env_file=None)
    with pytest.raises(Exception):
        s.model_agent = "x"


def test_rag_settings_defaults_when_yaml_absent(config_dir: Path):
    s = load_settings(config_dir=config_dir, env_file=None)
    assert s.embedding_model == "BAAI/bge-small-en-v1.5"
    assert s.rag_collection == "pa_guidance"
    assert s.rag_top_k == 4
    assert s.rag_min_score == 0.30
    assert s.rag_rewrite_min_score == 0.20
    assert s.rag_query_prefix == ""


def test_rag_settings_load_from_yaml(config_dir: Path):
    (config_dir / "models.yaml").write_text(
        "agent: gemini-flash-latest\n"
        "summarizer: gemini-flash-lite-latest\n"
        "temperature_agent: 0.0\n"
        "embedding_model: BAAI/bge-base-en-v1.5\n"
    )
    (config_dir / "rag.yaml").write_text(
        "collection: pa_guidance_v2\n"
        "top_k: 6\n"
        "min_score: 0.42\n"
        "rewrite_min_score: 0.25\n"
        'query_prefix: "Represent this sentence: "\n'
    )
    s = load_settings(config_dir=config_dir, env_file=None)
    assert s.embedding_model == "BAAI/bge-base-en-v1.5"
    assert s.rag_collection == "pa_guidance_v2"
    assert s.rag_top_k == 6
    assert s.rag_min_score == 0.42
    assert s.rag_rewrite_min_score == 0.25
    assert s.rag_query_prefix == "Represent this sentence: "


def test_memory_config_has_policy_defaults(config_dir: Path):
    (config_dir / "memory.yaml").write_text(
        "namespaces:\n"
        "  episodic: {ttl_days: 90, cap: 50}\n"
        "  member: {ttl_days: 365, cap: 100}\n"
        "  provider: {ttl_days: 365, cap: 100}\n"
        "  policy_notes: {ttl_days: null, cap: 200}\n"
        "importance_weights: {routine: 1.0, notable: 1.5, critical: 3.0}\n"
        "recency_half_life_days: 30\n"
    )
    m = load_settings(config_dir=config_dir, env_file=None).memory
    assert m.default_importance == "routine"
    assert isinstance(m.semantic_fields, tuple)
    assert "content" in m.semantic_fields and "text" in m.semantic_fields
    assert m.namespaces["episodic"].ttl_days == 90
    assert m.namespaces["policy_notes"].ttl_days is None


def test_memory_config_policy_fields_from_yaml(config_dir: Path):
    (config_dir / "memory.yaml").write_text(
        "namespaces:\n"
        "  episodic: {ttl_days: 90, cap: 50}\n"
        "importance_weights: {routine: 1.0}\n"
        "default_importance: notable\n"
        "semantic_fields: [body, note]\n"
    )
    m = load_settings(config_dir=config_dir, env_file=None).memory
    assert m.default_importance == "notable"
    assert m.semantic_fields == ("body", "note")


def test_rag_settings_env_overrides(config_dir: Path, monkeypatch):
    monkeypatch.setenv("PA_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    monkeypatch.setenv("PA_RAG_COLLECTION", "pa_guidance_env")
    monkeypatch.setenv("PA_RAG_TOP_K", "9")
    monkeypatch.setenv("PA_RAG_MIN_SCORE", "0.55")
    monkeypatch.setenv("PA_RAG_REWRITE_MIN_SCORE", "0.33")
    monkeypatch.setenv("PA_RAG_QUERY_PREFIX", "query: ")
    s = load_settings(config_dir=config_dir, env_file=None)
    assert s.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert s.rag_collection == "pa_guidance_env"
    assert s.rag_top_k == 9
    assert s.rag_min_score == 0.55
    assert s.rag_rewrite_min_score == 0.33
    assert s.rag_query_prefix == "query: "
