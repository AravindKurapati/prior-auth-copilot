"""AC-06: tiered memory. Tier 1 (working memory) recalls a fact stated on an
earlier simulated turn; Tier 2 (long-term semantic store) recalls a
prior-session record. Also guards the committed trace's byte-stability."""

import json
import sys
from pathlib import Path

import pytest

from pa_copilot.memory import policy, working
from pa_copilot.memory.store import memory_store

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from memory_demo import build_payload  # noqa: E402

TRACE = _REPO_ROOT / "traces" / "tiered_memory_recall.json"
NS = ("pa", "member", "M100001")


def test_working_memory_recalls_a_fact_from_an_earlier_turn():
    wm = {}
    wm = working.remember(wm, "conservative_care_weeks", 8)          # turn 1
    wm = working.remember(wm, "chief_complaint", "radicular pain")   # turn 2
    assert working.recall(wm, "conservative_care_weeks") == 8        # turn 3
    assert working.search_working(wm, "radicular") == [("chief_complaint", "radicular pain")]


def test_long_term_store_recalls_prior_record(tmp_path, fake_embedder):
    with memory_store(tmp_path / "m.db", embedder=fake_embedder) as store:
        store.put(NS, "det-2025-11", {
            "content": "Prior auth for lumbar MRI 72148 approved 2025-11; 8 weeks PT documented.",
            "importance": "notable",
        })
        hits = policy.search_ranked(store, NS, "history of lumbar MRI prior authorization", limit=3)
        assert hits and hits[0].key == "det-2025-11"
        assert "AC-06"  # id marker


def test_committed_trace_is_byte_stable(tmp_path, fake_embedder):
    with memory_store(tmp_path / "m.db", embedder=fake_embedder) as store:
        payload = build_payload(store)
        fresh = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if not store.semantic_index_available:
            pytest.skip(
                "sqlite-vec unavailable — byte-stability guard needs the semantic "
                "index that produced the committed trace"
            )
    assert fresh == TRACE.read_text(encoding="utf-8")
