"""AC-07 cross-session persistence, two real processes.

    python scripts/run_persistence_test.py                 # writes traces/memory_persistence.log
    python scripts/run_persistence_test.py --stdout-only    # print, don't write (test use)

Process 1 writes a determination record and exits. Process 2 is a brand-new
interpreter that opens a fresh store at the same path and recalls the record by
exact key -- no LLM, exact string match. Combined stdout (volatile timestamps
masked as <ts>, pids already printed as the literal <pid>) is the committed
evidence.

``sys.path[0]`` is ``scripts/`` when invoked this way, so put the repo root (and
``src``) on the path first (Ruling R3, same as ``scripts/agentic_rag_demo.py``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

NS = ("pa", "member", "M100001")
KEY = "det-cross-session"
REC = {
    "content": "Determination: lumbar MRI 72148 APPROVED; cited PA-MRI-LUMBAR step-therapy.",
    "importance": "critical",
    "disposition": "approve",
}
_TS = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?")


def _fake():
    """Bare-import the shared fake (Task 6/R32 ruling): the repo's ``tests/``
    is not a package, so this must be ``from _fakes import FakeEmbedder`` with
    ``tests/`` on ``sys.path`` -- NOT ``from tests._fakes import``. Each
    subprocess is a fresh interpreter re-running this module, so the insert
    happens here (call time), not just once at import time in the parent.
    """
    sys.path.insert(0, str(_REPO_ROOT / "tests"))
    from _fakes import FakeEmbedder  # noqa: PLC0415

    return FakeEmbedder()


def _pid() -> str:
    return "<pid>"  # never a real pid -- byte-stable by construction


def write_session(db_path: str) -> None:
    """Process A: open a store, write the determination record, close it."""
    from pa_copilot.memory.store import memory_store  # noqa: PLC0415

    with memory_store(db_path, embedder=_fake()) as store:
        store.put(NS, KEY, REC)
    print(f"[session-A pid={_pid()}] wrote {list(NS)}/{KEY}: {json.dumps(REC, sort_keys=True)}")


def read_session(db_path: str) -> int:
    """Process B: open a FRESH store at the same path, recall by exact key.

    Returns 0 on an exact-value match, 1 otherwise (miss or mismatch) -- this is
    the real contract the test asserts on, per the Global Constraints (the
    Interfaces section's `-> dict` sketch is superseded by this).
    """
    from pa_copilot.memory.store import memory_store  # noqa: PLC0415

    with memory_store(db_path, embedder=_fake()) as store:
        item = store.get(NS, KEY)
    if item is None:
        print(f"[session-B pid={_pid()}] MISS -- persistence FAILED")
        return 1
    ok = item.value == REC
    print(f"[session-B pid={_pid()}] recalled {list(NS)}/{KEY}: {json.dumps(item.value, sort_keys=True)}")
    print(f"[session-B pid={_pid()}] exact-match={ok}")
    return 0 if ok else 1


def _run_child(db_path: str, mode: str) -> tuple[str, int]:
    res = subprocess.run(
        [sys.executable, __file__, "--db", db_path, "--_child", mode],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0 and res.stderr:
        # Diagnostic-only: goes to the PARENT's own stderr so a human/CI log
        # shows the child's traceback. Never folded into `combined` (the
        # committed log) or the --stdout-only text the test diffs against --
        # both must stay byte-stable on the success path.
        print(res.stderr, file=sys.stderr, end="")
    return _TS.sub("<ts>", res.stdout), res.returncode


def _cleanup(db_path: str) -> None:
    for suffix in ("", "-wal", "-shm", "-journal"):
        Path(db_path + suffix).unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None, help="sqlite path; a fresh temp file is used if omitted")
    ap.add_argument("--stdout-only", action="store_true")
    ap.add_argument("--_child", choices=["write", "read"], help=argparse.SUPPRESS)
    args = ap.parse_args()

    # Child invocation: this same file, run as a brand-new interpreter by
    # `_run_child` above. Just do the one thing and exit.
    if args._child == "write":
        write_session(args.db)
        return 0
    if args._child == "read":
        return read_session(args.db)

    # Parent/orchestrator invocation: spawn the two real child processes.
    owns_db = args.db is None
    if owns_db:
        fd, db_path = tempfile.mkstemp(suffix=".db", prefix="memory_persistence_")
        os.close(fd)
    else:
        db_path = args.db
    _cleanup(db_path)  # start from a clean slate either way

    try:
        a_out, a_rc = _run_child(db_path, "write")
        b_out, b_rc = _run_child(db_path, "read")
    finally:
        _cleanup(db_path)

    passed = a_rc == 0 and b_rc == 0
    combined = (
        "# AC-07 cross-session memory persistence -- two python processes\n"
        f"{a_out}{b_out}"
        f"RESULT: {'PASS' if passed else 'FAIL'}\n"
    )

    if args.stdout_only:
        sys.stdout.write(combined)
    else:
        (_REPO_ROOT / "traces" / "memory_persistence.log").write_text(
            combined, encoding="utf-8", newline="\n"
        )
        print("wrote traces/memory_persistence.log")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
