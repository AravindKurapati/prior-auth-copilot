"""C1 guard: the RAG modules must import from a cwd outside the repo root.

`pa_copilot` ships only `packages.find(where=["src"])`; `data/` is a repo-root
package that exists on `sys.path` under pytest (`pythonpath=["."]`) but nowhere
else. PR5's `medical_necessity` worker, PR7's `pac` console script + Streamlit
app, and CI all import these modules from a non-repo-root cwd, so a stray
`from data.synthetic... import ...` would break every one of them.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    ["pa_copilot.rag.tool", "pa_copilot.rag.index", "pa_copilot.rag.corpus"],
)
def test_module_imports_from_outside_repo_root(module: str, tmp_path) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=str(tmp_path),  # a dir with no `data/` package on it
        env={**os.environ},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"`import {module}` failed from {tmp_path}:\n{proc.stderr}"
    )
