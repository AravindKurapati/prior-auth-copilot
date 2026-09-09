from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

TRACE_SCHEMA_VERSION = "1"
_MEMBER_ID_RE = re.compile(r"M\d{6,}", re.IGNORECASE)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_redact_values(data_dir: Path | None = None) -> list[str]:
    """Names + member-id keys that must be scrubbed from every trace by default.

    Reads `<data_dir>/synthetic/providers.json` (all `name` values) and
    `<data_dir>/synthetic/benefits.json` (all member-id keys). Missing or
    unreadable files are tolerated and contribute nothing.
    """
    base = Path(data_dir) if data_dir is not None else _REPO_ROOT / "data"
    values: list[str] = []

    try:
        providers = json.loads((base / "synthetic" / "providers.json").read_text(encoding="utf-8"))
        values.extend(
            v["name"] for v in providers.values() if isinstance(v, dict) and v.get("name")
        )
    except (OSError, ValueError):
        pass

    try:
        benefits = json.loads((base / "synthetic" / "benefits.json").read_text(encoding="utf-8"))
        values.extend(str(k) for k in benefits.keys())
    except (OSError, ValueError):
        pass

    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _stringify_keys(obj: Any) -> Any:
    """Recursively coerce any non-primitive dict key to `str(key)`.

    `json.dumps` never applies its `default` hook to keys, so tuple namespace
    keys (e.g. `("pa","member","M100001")`) would raise `TypeError` on write.
    """
    if isinstance(obj, dict):
        return {
            (k if isinstance(k, (str, bool, int, float)) or k is None else str(k)): _stringify_keys(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_stringify_keys(v) for v in obj]
    return obj


def redact(obj: Any, secrets: Iterable[str]) -> Any:
    secret_list = [s for s in secrets if s]

    def _scrub_str(s: str) -> str:
        for sec in secret_list:
            s = s.replace(sec, "<redacted>")
        return _MEMBER_ID_RE.sub("<redacted>", s)

    if isinstance(obj, str):
        return _scrub_str(obj)
    if isinstance(obj, dict):
        return {
            (_scrub_str(k) if isinstance(k, str) else k): redact(v, secret_list)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [redact(v, secret_list) for v in obj]
    return obj


class RunTracer:
    def __init__(
        self,
        trace_dir: str | Path,
        case_id: str,
        redact_values: list[str],
        session_id: str | None = None,
    ):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.case_id = case_id
        self.session_id = session_id
        merged = list(redact_values)
        for v in default_redact_values():
            if v not in merged:
                merged.append(v)
        self.redact_values = merged
        self.started_at = _now()
        self.events: list[dict] = []

    def event(self, node: str, kind: str, payload: dict) -> None:
        self.events.append({"ts": _now(), "node": node, "kind": kind, "payload": payload})

    def finish(self, decision: dict | None = None) -> Path:
        path = self.trace_dir / f"{self.case_id}.json"
        base = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "case_id": self.case_id,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "finished_at": _now(),
            "events": self.events,
            "decision": decision,
        }
        try:
            doc = _stringify_keys(base)
            doc = json.loads(json.dumps(doc, default=str))
            doc = redact(doc, self.redact_values)
            text = json.dumps(doc, indent=2)
        except Exception as exc:  # noqa: BLE001 - degrade, never lose the trace
            # Losing partial evidence to a serialization crash is worse than a
            # degraded trace — write what we can.
            degraded = redact(
                {
                    "schema_version": TRACE_SCHEMA_VERSION,
                    "case_id": self.case_id,
                    "session_id": self.session_id,
                    "error": repr(exc),
                    "events_count": len(self.events),
                },
                self.redact_values,
            )
            text = json.dumps(degraded, indent=2)
        path.write_text(text, encoding="utf-8", newline="\n")
        return path


def load_trace(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
