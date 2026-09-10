"""Clinical-guidance corpus loader + clause chunker.

Reads the synthetic `data/synthetic/clinical_guidance/*.md` narratives, splits each
into policy-tagged clause chunks (one per bullet / paragraph, plus the pre-heading
``lead``), and can dump them to a committed JSONL file for the index build.

Loading is deterministic: files are processed in sorted order and sections in
document order, with no randomness or wall-clock input.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from pa_copilot.config import load_settings

# filename stem -> (policy_id, service_code). Cross-checked against
# `data.synthetic.generators.SERVICES` by `test_every_policy_has_a_guidance_file`.
POLICY_BY_FILE: dict[str, tuple[str, str]] = {
    "pa-mri-lumbar": ("PA-MRI-LUMBAR", "72148"),
    "pa-knee-scope": ("PA-KNEE-SCOPE", "29881"),
    "pa-aflibercept": ("PA-AFLIBERCEPT", "J0178"),
    "pa-psg": ("PA-PSG", "95810"),
    "pa-egd": ("PA-EGD", "43239"),
    "pa-tfesi": ("PA-TFESI", "64483"),
}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    policy_id: str
    service_code: str
    doc_title: str
    section: str
    clause_index: int
    text: str


def _split_clauses(body_lines: list[str]) -> list[str]:
    """Split a section body into clauses: blank lines separate paragraphs and each
    ``- `` bullet is its own clause. Whitespace-stripped, empties dropped."""
    clauses: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            text = " ".join(part.strip() for part in buffer).strip()
            if text:
                clauses.append(text)
            buffer.clear()

    for line in body_lines:
        stripped = line.strip()
        if not stripped:
            flush()
        elif stripped.startswith("- "):
            flush()
            bullet = stripped[2:].strip()
            if bullet:
                clauses.append(bullet)
        else:
            buffer.append(line)
    flush()
    return clauses


def _parse_doc(text: str, policy_id: str, service_code: str) -> list[Chunk]:
    lines = text.splitlines()
    doc_title = ""
    if lines and lines[0].startswith("# "):
        doc_title = lines[0][2:].strip()
        lines = lines[1:]

    # (section_name, [body lines]) in document order; pre-heading block is "lead".
    sections: list[tuple[str, list[str]]] = [("lead", [])]
    for line in lines:
        if line.startswith("## "):
            sections.append((line[3:].strip(), []))
        else:
            sections[-1][1].append(line)

    chunks: list[Chunk] = []
    for section, body_lines in sections:
        section_slug = section.lower().replace(" ", "-")
        for clause_index, clause_text in enumerate(_split_clauses(body_lines)):
            chunks.append(
                Chunk(
                    chunk_id=f"{policy_id}:{section_slug}:{clause_index}",
                    policy_id=policy_id,
                    service_code=service_code,
                    doc_title=doc_title,
                    section=section,
                    clause_index=clause_index,
                    text=clause_text,
                )
            )
    return chunks


def load_guidance(guidance_dir: str | Path | None = None) -> list[Chunk]:
    """Load every guidance `*.md` into policy-tagged clause chunks.

    Defaults to ``<settings.synthetic_dir>/clinical_guidance``. Deterministic:
    files are read in sorted order, sections in document order.
    """
    if guidance_dir is None:
        guidance_dir = Path(load_settings(env_file=None).synthetic_dir) / "clinical_guidance"
    guidance_dir = Path(guidance_dir)

    chunks: list[Chunk] = []
    for path in sorted(guidance_dir.glob("*.md")):
        stem = path.stem
        if stem not in POLICY_BY_FILE:
            continue
        policy_id, service_code = POLICY_BY_FILE[stem]
        chunks.extend(_parse_doc(path.read_text(encoding="utf-8"), policy_id, service_code))
    return chunks


def chunks_to_rows(chunks: list[Chunk]) -> list[dict]:
    return [asdict(c) for c in chunks]


def write_chunks_jsonl(path: str | Path, chunks: list[Chunk]) -> None:
    rows = chunks_to_rows(chunks)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    Path(path).write_text(payload, encoding="utf-8", newline="\n")
