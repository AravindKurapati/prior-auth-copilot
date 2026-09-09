"""Regression trap for the pyproject `filterwarnings = ["error"]` policy.

If that line is ever weakened, a stray library warning silently re-enters the
suite. This test proves, in an isolated subprocess, that an emitted warning
still turns a run red.
"""

import subprocess
import sys


def test_emitted_warning_fails_the_run(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\nfilterwarnings = ["error"]\n', encoding="utf-8"
    )
    (tmp_path / "test_canary.py").write_text(
        "import warnings\n\n\ndef test_warns():\n    warnings.warn('boom', UserWarning)\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_canary.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0, result.stdout + result.stderr
    assert "UserWarning" in (result.stdout + result.stderr)
