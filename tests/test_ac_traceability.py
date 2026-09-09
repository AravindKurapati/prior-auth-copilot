"""Meta-test enforcing the AC-Traceability Rule across ledger updates.

For every row in `specs/acceptance-criteria.md` / `specs/nfr.md` whose Status is
`done` or `partial`, every `tests/test_*.py` file named in that row must exist.
"""

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_LEDGERS = [_ROOT / "specs" / "acceptance-criteria.md", _ROOT / "specs" / "nfr.md"]
_TEST_RE = re.compile(r"tests/test_[\w./-]+\.py")


def _table_rows(md_path: Path):
    for line in md_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3 or set("".join(cells)) <= {"-", ":"}:
            continue
        yield cells


def _tracked_test_files(md_path: Path) -> set[str]:
    wanted: set[str] = set()
    for cells in _table_rows(md_path):
        status = cells[-1].lower()
        if "done" not in status and "partial" not in status:
            continue
        wanted.update(_TEST_RE.findall(" ".join(cells)))
    return wanted


@pytest.mark.parametrize("ledger", _LEDGERS, ids=lambda p: p.name)
def test_named_tests_for_done_or_partial_rows_exist(ledger: Path):
    missing = sorted(rel for rel in _tracked_test_files(ledger) if not (_ROOT / rel).exists())
    assert not missing, f"{ledger.name} claims done/partial but these tests are absent: {missing}"


def test_meta_test_actually_finds_rows():
    # guard against the parser silently matching nothing
    assert any(_tracked_test_files(ledger) for ledger in _LEDGERS)
