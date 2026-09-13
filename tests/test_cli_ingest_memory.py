"""pac ingest / pac memory {list,search,show} / pac persistence-test — typer
CliRunner, no live LLM (none of these commands touch a model)."""

import asyncio
from dataclasses import dataclass

from typer.testing import CliRunner

from pa_copilot.cli import app

runner = CliRunner()


@dataclass
class _FakeSummary:
    doc_count: int = 1
    chunk_count: int = 1
    embedding_model: str = "fake"
    corpus_sha: str = "abc"
    collection: str = "guidance"


def test_ingest_builds_rag_index(monkeypatch):
    monkeypatch.setattr("pa_copilot.cli.build_index", lambda settings: _FakeSummary())
    monkeypatch.setattr("pa_copilot.cli.corpus.write_chunks_jsonl", lambda path, chunks: None)
    result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 0, result.output
    assert "RAG index built" in result.output


def test_ingest_fails_clearly_when_synthetic_data_missing(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PA_SYNTHETIC_DIR", str(empty))
    result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 1
    assert "Missing synthetic data files" in result.output


def test_memory_list_and_show_roundtrip(tmp_path, monkeypatch, fake_embedder):
    db = str(tmp_path / "mem.db")
    monkeypatch.setenv("PA_MEMORY_DB", db)
    from pa_copilot.memory.store import memory_store

    with memory_store(db, embedder=fake_embedder) as store:
        asyncio.run(store.aput(("pa", "member", "M1"), "det-1", {"content": "approved"}))

    result = runner.invoke(app, ["memory", "show", "pa,member,M1", "det-1"])
    assert result.exit_code == 0, result.output
    assert "approved" in result.output

    result = runner.invoke(app, ["memory", "list", "pa,member,M1"])
    assert result.exit_code == 0, result.output
    assert "det-1" in result.output


def test_memory_show_missing_key_exits_nonzero(tmp_path, monkeypatch, fake_embedder):
    db = str(tmp_path / "mem.db")
    monkeypatch.setenv("PA_MEMORY_DB", db)
    from pa_copilot.memory.store import memory_store

    with memory_store(db, embedder=fake_embedder):
        pass

    result = runner.invoke(app, ["memory", "show", "pa,member,M1", "nope"])
    assert result.exit_code == 1


def test_persistence_test_command_invokes_the_script(monkeypatch):
    calls = {}

    def fake_run(cmd, check):
        calls["cmd"] = cmd
        calls["check"] = check

    monkeypatch.setattr("pa_copilot.cli.subprocess.run", fake_run)
    result = runner.invoke(app, ["persistence-test"])
    assert result.exit_code == 0, result.output
    assert calls["check"] is True
    assert "run_persistence_test.py" in calls["cmd"][-1]
