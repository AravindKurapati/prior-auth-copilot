"""In-repo producer of traces/tiered_memory_recall.json (AC-06 evidence).

    python scripts/memory_demo.py

Deterministic: the FakeEmbedder (no model download) + no timestamps in the
payload, so the committed bytes are stable across runs. tests/test_ac06_tiered_memory.py
regenerates and diffs."""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from pa_copilot.memory import policy, working  # noqa: E402
from pa_copilot.memory.store import memory_store  # noqa: E402

NS = ("pa", "member", "M100001")


def build_payload(store) -> dict:
    # Tier 1 — working memory across simulated turns
    wm = {}
    wm = working.remember(wm, "conservative_care_weeks", 8)
    wm = working.remember(wm, "chief_complaint", "radicular pain")
    turn3 = working.recall(wm, "conservative_care_weeks")

    # Tier 2 — long-term semantic store
    store.put(NS, "det-2025-11", {
        "content": "Prior auth for lumbar MRI 72148 approved 2025-11; 8 weeks PT documented.",
        "importance": "notable",
    })
    hits = policy.search_ranked(store, NS, "history of lumbar MRI prior authorization", limit=1)
    top = hits[0]

    return {
        "ac": "AC-06",
        "scenario": "MRI lumbar re-request for member M100001; prior determination recalled",
        "tier_1_working_memory": {
            "writes": [
                {"turn": 1, "key": "conservative_care_weeks", "value": 8},
                {"turn": 2, "key": "chief_complaint", "value": "radicular pain"},
            ],
            "turn_3_recall": {"key": "conservative_care_weeks", "recalled": turn3},
        },
        "tier_2_long_term": {
            "namespace": list(NS),
            "written_key": "det-2025-11",
            "query": "history of lumbar MRI prior authorization",
            "top_hit_key": top.key,
            "top_hit_content": top.value["content"],
            "top_hit_importance": top.value["importance"],
        },
        "embedder": "FakeEmbedder",
        "semantic_index": store.semantic_index_available,
    }


def main() -> None:
    out = _REPO_ROOT / "traces" / "tiered_memory_recall.json"
    db_path = _REPO_ROOT / ".pa_memory_demo.db"
    with memory_store(db_path, embedder=_fake()) as store:
        payload = build_payload(store)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")
    db_path.unlink(missing_ok=True)
    print(out.relative_to(_REPO_ROOT).as_posix())


def _fake():
    sys.path.insert(0, str(_REPO_ROOT / "tests"))
    from _fakes import FakeEmbedder

    return FakeEmbedder()


if __name__ == "__main__":
    main()
