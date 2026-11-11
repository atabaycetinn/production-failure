from pathlib import Path

import pytest

from mlrca.config import Settings
from mlrca.eval import build_context

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "data" / "scenarios" / "fraud_precision_drop"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(corpus_dir=ROOT / "data" / "corpus", scenarios_dir=ROOT / "data" / "scenarios", runs_dir=tmp_path,
                    fallbacks_enabled=True)


@pytest.fixture
def ctx(settings):
    return build_context(SCENARIO, settings)
