"""NFR-01: no secrets or API keys committed; env-var config with `.env.example`."""

import re
import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

_KEY_SHAPES = [
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),          # Google API key
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"),       # OpenAI-style secret key (incl. sk-proj-/sk-ant-api03-)
    re.compile(
        r"(?:GEMINI|GOOGLE)_API_KEY['\"]?\]?[ \t]*[=:][ \t]*['\"]?"
        r"(?!<|\$|\.\.\.|your[-_]|xxx|placeholder|dummy|fake|test[-_]|example)"
        r"[A-Za-z0-9_\-]{20,}"
    ),  # assigned realistic key value: KEY = "...", KEY: ..., os.environ["KEY"] = "..."
]

_SKIP_SUFFIXES = {".png", ".db", ".ico", ".jpg", ".jpeg", ".gz", ".zip"}


def _tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=_ROOT, capture_output=True, text=True, check=True
    )
    return [line for line in out.stdout.splitlines() if line]


def test_no_committed_dotenv():
    assert ".env" not in _tracked_files()
    assert (_ROOT / ".env.example").is_file()


def test_gitignore_excludes_dotenv():
    assert ".env" in (_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()


def test_no_key_shaped_strings_in_tracked_text():
    offenders: list[str] = []
    for rel in _tracked_files():
        if rel == "tests/test_nfr01_no_secrets.py":
            continue  # this file names the patterns on purpose
        p = _ROOT / rel
        if p.suffix.lower() in _SKIP_SUFFIXES or not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for pat in _KEY_SHAPES:
            if pat.search(text):
                offenders.append(f"{rel}: {pat.pattern}")
    assert not offenders, offenders
