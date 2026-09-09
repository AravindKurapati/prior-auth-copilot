"""Regenerate all synthetic corpora + sample requests. Run from repo root:

    python scripts/make_samples.py

When run this way, sys.path[0] is `scripts/`, not the repo root, so
`from data.synthetic import generators` would fail. Put the repo root on the path
first (Ruling R3).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.synthetic import generators  # noqa: E402


def main() -> None:
    generators.write_all(Path(__file__).resolve().parents[1])
    print("wrote data/synthetic/* and data/samples/*")


if __name__ == "__main__":
    main()
