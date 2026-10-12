"""Runtime configuration, loaded from environment with sane defaults.

Everything the grader might want to change is an env var with a working default,
so the app boots with zero configuration beyond an API key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    # --- LLM ---
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )
    model: str = field(default_factory=lambda: os.getenv("MODEL", "claude-opus-5"))
    max_tokens: int = field(default_factory=lambda: _env_int("MAX_TOKENS", 2000))
    effort: str = field(default_factory=lambda: os.getenv("EFFORT", "medium"))

    # --- Retrieval ---
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 1200))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 200))
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 6))
    # Below this combined relevance score, the question is treated as out of
    # scope before we ever spend a token on the LLM. See app/core/guard.py for
    # how the score is computed and docs/scope-tuning.md for how to tune it.
    relevance_floor: float = field(
        default_factory=lambda: _env_float("RELEVANCE_FLOOR", 0.40)
    )

    # --- Storage ---
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DATA_DIR", str(REPO_ROOT / "data")))
    )

    # --- Upload limits ---
    max_upload_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 25))

    # --- Behaviour ---
    # When True, the bot refuses anything not grounded in the uploaded corpus.
    strict_scope: bool = field(default_factory=lambda: _env_bool("STRICT_SCOPE", True))

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def has_api_key(self) -> bool:
        return bool(self.anthropic_api_key.strip())

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "uploads").mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Process-wide settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings


def reset_settings() -> None:
    """Drop the cached settings (used by tests that tweak the environment)."""
    global _settings
    _settings = None
