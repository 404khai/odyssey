"""Configuration loading for Odyssey experiments.

Why this exists:
    Research runs must be reproducible. Loading a single YAML file as the
    source of truth keeps hyperparameters out of ad-hoc script constants.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to a YAML config. Defaults to ``configs/default.yaml``.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the root YAML value is not a mapping.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}")
    return data
