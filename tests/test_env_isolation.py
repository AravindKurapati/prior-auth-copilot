"""The autouse _env_snapshot fixture (tests/conftest.py) must revert any raw
os.environ mutation a test makes — monkeypatch alone cannot, and config.load_settings
does a raw os.environ write. Two tests in definition order: the first leaks, the
second proves the leak was reverted."""

import os


def test_aaa_writes_a_raw_environ_key():
    os.environ["PA_TEST_LEAK_CHECK"] = "leaked"
    assert os.environ["PA_TEST_LEAK_CHECK"] == "leaked"


def test_bbb_raw_environ_key_was_reverted():
    assert "PA_TEST_LEAK_CHECK" not in os.environ
