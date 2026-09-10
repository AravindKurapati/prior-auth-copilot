import os
from pathlib import Path

import pytest

from pa_copilot.config import get_settings


@pytest.fixture(autouse=True)
def _env_snapshot():
    """Snapshot os.environ and restore it exactly after each test. monkeypatch
    reverts its own setenv/delenv, but not a raw `os.environ[...] = ...` write
    (config.load_settings mirrors GEMINI_API_KEY -> GOOGLE_API_KEY that way)."""
    saved = dict(os.environ)
    try:
        yield
    finally:
        for key in [k for k in os.environ if k not in saved]:
            del os.environ[key]
        for key, value in saved.items():
            if os.environ.get(key) != value:
                os.environ[key] = value


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """`get_settings()` is an `lru_cache`d singleton; clear it around every test so
    config / memory-policy mutations in one test can't leak into the next."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def tmp_trace_dir(tmp_path: Path) -> Path:
    d = tmp_path / "traces"
    d.mkdir()
    return d


@pytest.fixture
def frozen_now() -> str:
    return "2026-09-09T12:00:00+00:00"


@pytest.fixture
def sample_request() -> dict:
    return {
        "case_id": "case-0001",
        "session_id": "sess-0001",
        "member_id": "M100001",
        "raw_provider_text": (
            "Requesting prior auth for MRI lumbar spine (72148). Patient has had 8 weeks "
            "of physical therapy and persistent radicular pain. DX M54.16."
        ),
        "structured": {
            "service_code": "72148",
            "diagnosis_codes": ["M54.16"],
            "requested_units": 1,
            "place_of_service": "outpatient",
            "provider_npi": "1093817465",
        },
    }
