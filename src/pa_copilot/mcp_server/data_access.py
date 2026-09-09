"""Pure domain lookups over the synthetic corpora.

Every function returns a plain, JSON-serializable dict and **never raises on a
miss** — a lookup that finds nothing returns a dict with ``found: False`` and a
human-readable ``note``. `server.py` wraps these as MCP tools/resources, and the
agent nodes in PR5 branch on the pinned keys (notably ``criteria_check`` returning
one of ``not_found`` / ``excluded`` / ``indeterminate``).

The three corpus files are immutable at runtime, so `load_corpora` caches the
parsed JSON at module level keyed by the resolved `synthetic_dir`.
"""

from __future__ import annotations

import json
from pathlib import Path

from pa_copilot.config import get_settings

_CORPORA_CACHE: dict[str, dict] = {}


def _synthetic_dir(synthetic_dir: str | Path | None) -> Path:
    if synthetic_dir is None:
        synthetic_dir = get_settings().synthetic_dir
    return Path(synthetic_dir)


def clear_corpora_cache() -> None:
    """Drop the module-level parsed-corpus cache (test ergonomics)."""
    _CORPORA_CACHE.clear()


def load_corpora(synthetic_dir: str | Path | None = None) -> dict:
    """Read `benefits.json`, `providers.json`, `criteria.json` once.

    Returns ``{"benefits": ..., "providers": ..., "criteria": ...}``. Cached at
    module level keyed by the resolved directory path.
    """
    base = _synthetic_dir(synthetic_dir).resolve()
    key = str(base)
    cached = _CORPORA_CACHE.get(key)
    if cached is None:
        cached = {
            "benefits": json.loads((base / "benefits.json").read_text(encoding="utf-8")),
            "providers": json.loads((base / "providers.json").read_text(encoding="utf-8")),
            "criteria": json.loads((base / "criteria.json").read_text(encoding="utf-8")),
        }
        _CORPORA_CACHE[key] = cached
    return cached


def _corpora(corpora: dict | None) -> dict:
    return corpora if corpora is not None else load_corpora()


def benefit_lookup(
    member_id: str, service_code: str, *, corpora: dict | None = None
) -> dict:
    """Whether ``member_id``'s plan covers ``service_code`` and needs prior auth."""
    benefits = _corpora(corpora)["benefits"]
    member = benefits.get(member_id)
    if member is None:
        return {
            "found": False,
            "member_id": member_id,
            "service_code": service_code,
            "plan_id": None,
            "covered": False,
            "requires_pa": False,
            "network_status": None,
            "note": "member not found",
        }

    entry = member.get("covered_services", {}).get(service_code)
    if entry is None:
        return {
            "found": False,
            "member_id": member_id,
            "service_code": service_code,
            "plan_id": member.get("plan_id"),
            "covered": False,
            "requires_pa": False,
            "network_status": None,
            "note": "service not covered under plan",
        }

    covered = bool(entry.get("covered", False))
    return {
        "found": True,
        "member_id": member_id,
        "service_code": service_code,
        "plan_id": member.get("plan_id"),
        "covered": covered,
        "requires_pa": bool(entry.get("requires_pa", False)),
        "network_status": entry.get("network_status"),
        "note": (
            "service covered under plan"
            if covered
            else "service listed on plan but not covered"
        ),
    }


def provider_lookup(npi: str, *, corpora: dict | None = None) -> dict:
    """Look up an ordering provider by NPI."""
    providers = _corpora(corpora)["providers"]
    entry = providers.get(npi)
    if entry is None:
        return {
            "found": False,
            "npi": npi,
            "name": None,
            "specialty": None,
            "network_status": None,
            "note": "provider not found",
        }
    return {
        "found": True,
        "npi": npi,
        "name": entry.get("name"),
        "specialty": entry.get("specialty"),
        "network_status": entry.get("network_status"),
        "note": "provider found",
    }


def _find_policy_by_service(criteria: dict, service_code: str) -> tuple[str | None, dict | None]:
    for policy_id, policy in criteria.items():
        if policy.get("service_code") == service_code:
            return policy_id, policy
    return None, None


def criteria_check(
    service_code: str, diagnosis_codes: list[str], *, corpora: dict | None = None
) -> dict:
    """Mechanical medical-necessity check for ``service_code``.

    ``status`` is one of ``not_found`` (no policy for the service code),
    ``excluded`` (a supplied diagnosis is a documented exclusion), or
    ``indeterminate`` (policy exists but only a human can verify the conditions).
    """
    criteria = _corpora(corpora)["criteria"]
    policy_id, policy = _find_policy_by_service(criteria, service_code)
    if policy is None:
        return {
            "found": False,
            "policy_id": None,
            "title": None,
            "required_conditions": [],
            "exclusions": [],
            "evidence_requirements": [],
            "excluded_diagnoses": [],
            "status": "not_found",
            "note": f"no prior-auth policy found for service code {service_code}",
        }

    excluded_diagnoses = list(policy.get("excluded_diagnoses", []))
    base = {
        "found": True,
        "policy_id": policy_id,
        "title": policy.get("title"),
        "required_conditions": list(policy.get("required_conditions", [])),
        "exclusions": list(policy.get("exclusions", [])),
        "evidence_requirements": list(policy.get("evidence_requirements", [])),
        "excluded_diagnoses": excluded_diagnoses,
    }

    for code in (diagnosis_codes or []):
        if code in excluded_diagnoses:
            return {
                **base,
                "status": "excluded",
                "note": f"diagnosis {code} is a documented exclusion for this policy",
            }

    return {
        **base,
        "status": "indeterminate",
        "note": (
            "policy criteria require clinical assessment; mechanical check cannot "
            "verify conditions"
        ),
    }


def get_policy(policy_id: str, *, corpora: dict | None = None) -> dict:
    """Return the full policy dict for ``policy_id`` (with ``found``/``policy_id``)."""
    criteria = _corpora(corpora)["criteria"]
    policy = criteria.get(policy_id)
    if policy is None:
        return {"found": False, "policy_id": policy_id}
    return {"found": True, "policy_id": policy_id, **policy}


def list_policies(*, corpora: dict | None = None) -> dict:
    """List every policy as ``{policy_id, service_code, title}``."""
    criteria = _corpora(corpora)["criteria"]
    return {
        "policies": [
            {
                "policy_id": policy_id,
                "service_code": policy.get("service_code"),
                "title": policy.get("title"),
            }
            for policy_id, policy in criteria.items()
        ]
    }
