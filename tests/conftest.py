import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_data_config  # noqa: E402
from src.constants import PROCESSED_DIR  # noqa: E402


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_data_config()


@pytest.fixture(scope="session")
def panel(cfg) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{cfg['version']}.parquet"
    if not path.exists():
        pytest.skip(f"{path.name} not built; run `python -m src.cli features` first")
    return pd.read_parquet(path)
