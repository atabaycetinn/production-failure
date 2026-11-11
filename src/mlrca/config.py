"""Runtime configuration, overridable through environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    model: str = field(default_factory=lambda: os.getenv("MLRCA_MODEL", "claude-opus-5-5"))
    effort: str = field(default_factory=lambda: os.getenv("MLRCA_EFFORT", "high"))
    max_tokens: int = 16000
    max_turns: int = int(os.getenv("MLRCA_MAX_TURNS", "40"))
    corpus_dir: Path = field(
        default_factory=lambda: Path(os.getenv("MLRCA_CORPUS_DIR", PROJECT_ROOT / "data" / "corpus"))
    )
    scenarios_dir: Path = field(
        default_factory=lambda: Path(os.getenv("MLRCA_SCENARIOS_DIR", PROJECT_ROOT / "data" / "scenarios"))
    )
    runs_dir: Path = field(default_factory=lambda: Path(os.getenv("MLRCA_RUNS_DIR", PROJECT_ROOT / "runs")))
    # Server-side refusal fallback for Claude Opus 5.5; set MLRCA_FALLBACKS=off to disable.
    fallbacks_enabled: bool = field(default_factory=lambda: os.getenv("MLRCA_FALLBACKS", "on") != "off")


def get_settings() -> Settings:
    return Settings()
