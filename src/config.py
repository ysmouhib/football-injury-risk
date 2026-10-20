"""YAML config loading and content hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .constants import CONFIGS_DIR

DATA_CONFIG = CONFIGS_DIR / "data.yaml"
LABEL_MAPPING_CONFIG = CONFIGS_DIR / "label_mapping.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_data_config(path: Path = DATA_CONFIG) -> dict[str, Any]:
    cfg = load_yaml(path)
    # YAML keys are ints already; normalise override dates to strings for hashing.
    cfg["season_end_overrides"] = {
        int(k): str(v) for k, v in (cfg.get("season_end_overrides") or {}).items()
    }
    return cfg


def stable_hash(obj: Any) -> str:
    """sha256 of a JSON-serialisable object, independent of key order."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
