from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class NamespacePolicy:
    ttl_days: int | None
    cap: int


@dataclass(frozen=True)
class MemoryConfig:
    namespaces: dict[str, NamespacePolicy]
    importance_weights: dict[str, float]
    recency_half_life_days: int
    default_importance: str = "routine"
    semantic_fields: tuple[str, ...] = ("content", "text")
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str | None
    model_agent: str
    model_summarizer: str
    temperature_agent: float
    state_db: str
    memory_db: str
    chroma_dir: str
    repo_root: str
    data_dir: str
    samples_dir: str
    synthetic_dir: str
    traces_dir: str
    embedding_model: str
    rag_collection: str
    rag_top_k: int
    rag_min_score: float
    rag_rewrite_min_score: float
    rag_query_prefix: str
    max_replans: int
    max_hops: int
    recursion_limit: int
    tau: float
    memory: MemoryConfig


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _build_memory_config(raw: dict) -> MemoryConfig:
    namespaces = {
        name: NamespacePolicy(ttl_days=pol.get("ttl_days"), cap=int(pol.get("cap", 0)))
        for name, pol in (raw.get("namespaces") or {}).items()
    }
    weights = {k: float(v) for k, v in (raw.get("importance_weights") or {}).items()}
    return MemoryConfig(
        namespaces=namespaces,
        importance_weights=weights,
        recency_half_life_days=int(raw.get("recency_half_life_days", 30)),
        default_importance=str(raw.get("default_importance", "routine")),
        semantic_fields=tuple(raw.get("semantic_fields") or ("content", "text")),
        raw=raw,
    )


def load_settings(
    config_dir: str | Path = "config",
    env_file: str | Path | None = ".env",
) -> Settings:
    if env_file and Path(env_file).exists():
        load_dotenv(env_file)

    # The project standardises on GEMINI_API_KEY (see .env.example), but
    # langchain-google-genai (`init_chat_model("google_genai:…")` /
    # ChatGoogleGenerativeAI) only reads GOOGLE_API_KEY. Mirror it so a machine
    # configured per our docs actually authenticates.
    if os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

    cfg = Path(config_dir)
    models = _read_yaml(cfg / "models.yaml")
    routing = _read_yaml(cfg / "routing.yaml")
    memory = _read_yaml(cfg / "memory.yaml")
    rag = _read_yaml(cfg / "rag.yaml")

    repo_root = os.environ.get("PA_REPO_ROOT") or str(_REPO_ROOT)
    data_dir = os.environ.get("PA_DATA_DIR") or str(Path(repo_root) / "data")
    samples_dir = os.environ.get("PA_SAMPLES_DIR") or str(Path(data_dir) / "samples")
    synthetic_dir = os.environ.get("PA_SYNTHETIC_DIR") or str(Path(data_dir) / "synthetic")
    traces_dir = os.environ.get("PA_TRACES_DIR") or str(Path(repo_root) / "traces")

    return Settings(
        gemini_api_key=os.environ.get("GEMINI_API_KEY"),
        model_agent=os.environ.get("PA_MODEL_AGENT", models.get("agent", "gemini-flash-latest")),
        model_summarizer=os.environ.get(
            "PA_MODEL_SUMMARIZER", models.get("summarizer", "gemini-flash-lite-latest")
        ),
        temperature_agent=float(
            os.environ.get("PA_TEMPERATURE_AGENT", models.get("temperature_agent", 0.0))
        ),
        state_db=os.environ.get("PA_STATE_DB", "./.pa_state.db"),
        memory_db=os.environ.get("PA_MEMORY_DB", "./.pa_memory.db"),
        chroma_dir=os.environ.get("PA_CHROMA_DIR", "./.pa_chroma"),
        repo_root=repo_root,
        data_dir=data_dir,
        samples_dir=samples_dir,
        synthetic_dir=synthetic_dir,
        traces_dir=traces_dir,
        embedding_model=os.environ.get(
            "PA_EMBEDDING_MODEL", models.get("embedding_model", "BAAI/bge-small-en-v1.5")
        ),
        rag_collection=os.environ.get(
            "PA_RAG_COLLECTION", rag.get("collection", "pa_guidance")
        ),
        rag_top_k=int(os.environ.get("PA_RAG_TOP_K", rag.get("top_k", 4))),
        rag_min_score=float(os.environ.get("PA_RAG_MIN_SCORE", rag.get("min_score", 0.30))),
        rag_rewrite_min_score=float(
            os.environ.get("PA_RAG_REWRITE_MIN_SCORE", rag.get("rewrite_min_score", 0.20))
        ),
        rag_query_prefix=os.environ.get(
            "PA_RAG_QUERY_PREFIX", rag.get("query_prefix", "")
        ),
        max_replans=int(routing.get("max_replans", 2)),
        max_hops=int(routing.get("max_hops", 12)),
        recursion_limit=int(routing.get("recursion_limit", 40)),
        tau=float(routing.get("tau", 0.55)),
        memory=_build_memory_config(memory),
    )


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
