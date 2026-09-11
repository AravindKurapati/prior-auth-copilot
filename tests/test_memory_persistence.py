"""AC-07: memory persists across sessions. Form 1 — in-process: session A writes,
all objects torn down, session B opens a FRESH store at the same path and recalls
by exact key (no LLM). Form 2 — two real `python` processes via the committed
script; its combined stdout is the committed traces/memory_persistence.log."""

import importlib.util
import subprocess
import sys
from pathlib import Path

from pa_copilot.memory.store import memory_store

REPO = Path(__file__).resolve().parents[1]
LOG = REPO / "traces" / "memory_persistence.log"
NS = ("pa", "member", "M100001")
REC = {"content": "Determination: lumbar MRI 72148 APPROVED; cited PA-MRI-LUMBAR step-therapy.",
       "importance": "critical", "disposition": "approve"}


def _load_run_persistence_module():
    """Load scripts/run_persistence_test.py by path (it's a script, not a package
    member) so `_run_child` can be unit-tested directly without a real subprocess."""
    path = REPO / "scripts" / "run_persistence_test.py"
    spec = importlib.util.spec_from_file_location("run_persistence_test_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_in_process_teardown_then_fresh_store_recalls(tmp_path, fake_embedder):
    db = tmp_path / "persist.db"
    with memory_store(db, embedder=fake_embedder) as a:
        a.put(NS, "det-1", REC)
    # `a` and its connection are closed here.
    with memory_store(db, embedder=fake_embedder) as b:
        got = b.get(NS, "det-1")
    assert got is not None
    assert got.value["disposition"] == "approve"
    assert got.value["content"] == REC["content"]


def test_committed_log_matches_a_fresh_two_process_run(tmp_path):
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "run_persistence_test.py"),
         "--db", str(tmp_path / "x.db"), "--stdout-only"],
        capture_output=True, text=True, check=True,
    )
    assert out.stdout == LOG.read_text(encoding="utf-8")


def test_run_child_surfaces_crash_stderr_without_polluting_output(monkeypatch, capsys):
    """Task 8 review finding (Important): a crashed child's stderr must reach the
    PARENT's own stderr for diagnosability, but never leak into the byte-stable
    text (`combined` / --stdout-only) the committed log is diffed against."""
    mod = _load_run_persistence_module()

    class _FakeCompletedProcess:
        stdout = "[session-A pid=<pid>] partial or no output\n"
        stderr = "Traceback (most recent call last):\nBoom: child crashed\n"
        returncode = 1

    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: _FakeCompletedProcess())

    out_text, rc = mod._run_child("unused.db", "write")

    captured = capsys.readouterr()
    assert rc == 1
    assert "Boom: child crashed" in captured.err  # surfaced to parent's own stderr
    assert "Boom: child crashed" not in out_text  # never in the byte-stable text
    assert "Traceback" not in out_text
