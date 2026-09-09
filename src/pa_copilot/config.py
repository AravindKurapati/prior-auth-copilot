from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str | None
    model_agent: str
    model_summarizer: str
    temperature_agent: float
    state_db: str
    memory_db: str
    chroma_dir: str
    max_replans: int
    max_hops: int
    recursion_limit: int
    tau: float
    memory: dict


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def load_settings(
    config_dir: str | Path = "config",
    env_file: str | Path | None = ".env",
) -> Settings:
    if env_file and Path(env_file).exists():
        load_dotenv(env_file)

    cfg = Path(config_dir)
    models = _read_yaml(cfg / "models.yaml")
    routing = _read_yaml(cfg / "routing.yaml")
    memory = _read_yaml(cfg / "memory.yaml")

    return Settings(
        gemini_api_key=os.environ.get("GEMINI_API_KEY"),
        model_agent=os.environ.get("PA_MODEL_AGENT", models.get("agent", "gemini-flash-latest")),
        model_summarizer=os.environ.get(
            "PA_MODEL_SUMMARIZER", models.get("summarizer", "gemini-flash-lite-latest")
        ),
        temperature_agent=float(models.get("temperature_agent", 0.0)),
        state_db=os.environ.get("PA_STATE_DB", "./.pa_state.db"),
        memory_db=os.environ.get("PA_MEMORY_DB", "./.pa_memory.db"),
        chroma_dir=os.environ.get("PA_CHROMA_DIR", "./.pa_chroma"),
        max_replans=int(routing.get("max_replans", 2)),
        max_hops=int(routing.get("max_hops", 12)),
        recursion_limit=int(routing.get("recursion_limit", 40)),
        tau=float(routing.get("tau", 0.55)),
        memory=memory,
    )


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
