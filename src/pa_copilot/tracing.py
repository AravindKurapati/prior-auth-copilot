from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

TRACE_SCHEMA_VERSION = "1"
_MEMBER_ID_RE = re.compile(r"M\d{6,}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    def __init__(self, trace_dir: str | Path, case_id: str, redact_values: list[str]):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.case_id = case_id
        self.redact_values = redact_values
        self.started_at = _now()
        self.events: list[dict] = []

    def event(self, node: str, kind: str, payload: dict) -> None:
        self.events.append({"ts": _now(), "node": node, "kind": kind, "payload": payload})

    def finish(self, decision: dict | None = None) -> Path:
        doc = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "case_id": self.case_id,
            "started_at": self.started_at,
            "finished_at": _now(),
            "events": self.events,
            "decision": decision,
        }
        # Normalize to JSON-safe primitives before redacting to catch non-primitive values
        doc = json.loads(json.dumps(doc, default=str))
        doc = redact(doc, self.redact_values)
        path = self.trace_dir / f"{self.case_id}.json"
        path.write_text(json.dumps(doc, indent=2))
        return path


def load_trace(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
