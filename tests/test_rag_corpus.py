import json
from pathlib import Path

from data.synthetic import generators as g
from pa_copilot.config import get_settings
from pa_copilot.rag import corpus


def test_every_policy_has_a_guidance_file():
    mapped = {(pid, sc) for pid, sc in corpus.POLICY_BY_FILE.values()}
    assert mapped == {(s["policy_id"], s["service_code"]) for s in g.SERVICES}


def test_load_is_deterministic_and_tagged():
    a = corpus.load_guidance()
    b = corpus.load_guidance()
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len(a) >= 24  # 6 docs, >= 4 clauses each
    psg = [c for c in a if c.policy_id == "PA-PSG"]
    assert psg and all(c.service_code == "95810" for c in psg)
    assert {c.section for c in psg} >= {"Indications", "Step therapy", "Exclusions"}


def test_chunk_ids_unique():
    ids = [c.chunk_id for c in corpus.load_guidance()]
    assert len(ids) == len(set(ids))


def test_committed_chunks_jsonl_matches_loader():
    """The committed dump must equal `chunks_to_rows(load_guidance())` line for
    line — guards against silent rot when a guidance `.md` is edited (I6).
    Regenerate with `make chunks` (or `python scripts/ingest_rag.py`)."""
    path = Path(get_settings().data_dir) / "synthetic" / "clinical_guidance_chunks.jsonl"
    committed = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert committed == corpus.chunks_to_rows(corpus.load_guidance())
