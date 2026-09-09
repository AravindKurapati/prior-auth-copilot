from pathlib import Path

import pytest


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
