"""Shared helpers for evaluation entry scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from marl_lisl.baselines import (
    GreedyConflictAwarePolicy,
    MaintainUntilConflictPolicy,
    ProactiveRuleBaseline,
    ShortestDelayPolicy,
)
from marl_lisl.utils.config import load_yaml


def resolve_project_path(project_root: Path, path: Path) -> Path:
    """Resolve a possibly relative path against the project root."""
    return path if path.is_absolute() else project_root / path


def load_env_config_for_traffic(
    config_path: Path,
    traffic_path: Path,
    project_root: Path,
    preload_graphs: bool = False,
) -> dict:
    """Load env config, inject a specific traffic file, and fix all data paths."""
    config = load_yaml(config_path)
    traffic_path = resolve_project_path(project_root, traffic_path)
    if not traffic_path.is_file():
        raise FileNotFoundError(f"Traffic file not found: {traffic_path}")
    for key in ("graph_dir", "traffic_dir", "traffic_train_path", "traffic_eval_path", "traffic_stress_path"):
        if key in config:
            config[key] = resolve_project_path(project_root, Path(config[key]))
    config["traffic_path"] = traffic_path
    config["num_flows"] = int(np.load(traffic_path).shape[0])
    config["graph_preload"] = bool(preload_graphs)
    candidates_cfg = dict(config.get("candidates", {}))
    for key in ("train_dir", "eval_dir", "stress_dir"):
        if key in candidates_cfg:
            candidates_cfg[key] = resolve_project_path(project_root, Path(candidates_cfg[key]))
    config["candidates"] = candidates_cfg
    if candidates_cfg.get("enabled", False):
        if traffic_path == Path(config["traffic_train_path"]):
            config["candidate_dir"] = candidates_cfg["train_dir"]
        elif traffic_path == Path(config["traffic_eval_path"]):
            config["candidate_dir"] = candidates_cfg["eval_dir"]
        elif "traffic_stress_path" in config and traffic_path == Path(config["traffic_stress_path"]):
            config["candidate_dir"] = candidates_cfg.get("stress_dir")
        else:
            config["candidates"]["enabled"] = False
            print("WARNING: custom traffic has no precomputed candidates; using online PathGenerator.")
    config["future_mutex"] = dict(config["future_mutex"])
    config["future_mutex"]["node_mutex_path"] = resolve_project_path(
        project_root, Path(config["future_mutex"]["node_mutex_path"])
    )
    return config


def baseline_policies(config: dict) -> list[tuple[str, object]]:
    """Create all baseline policies."""
    return [
        ("ShortestDelay", ShortestDelayPolicy()),
        ("MaintainUntilConflict", MaintainUntilConflictPolicy()),
        ("GreedyConflictAware", GreedyConflictAwarePolicy()),
        ("ProactiveRule", ProactiveRuleBaseline.from_config(config)),
    ]
