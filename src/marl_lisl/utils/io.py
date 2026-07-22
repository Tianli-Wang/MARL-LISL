"""Configuration and path I/O helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping with a clear dependency/error message."""
    try:
        import yaml  # type: ignore[import-untyped]  # PyYAML 官方包未内置类型声明
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyYAML is required; install it with: pip install pyyaml") from exc
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return config


def resolve_raw_dir(primary: Path, fallback: Path) -> Path:
    """Prefer the normalized raw-data location, then use its compatibility fallback."""
    if primary.is_dir():
        return primary
    if fallback.is_dir():
        return fallback
    raise FileNotFoundError(f"Neither raw directory exists: {primary}, {fallback}")
