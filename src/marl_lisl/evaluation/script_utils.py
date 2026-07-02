"""Shared helpers for evaluation entry scripts."""

from __future__ import annotations

from pathlib import Path

from marl_lisl.baselines import (
    GreedyConflictAwarePolicy,
    MaintainUntilConflictPolicy,
    ProactiveRuleBaseline,
    RSMRPolicy,
    ShortestDelayPolicy,
)
from marl_lisl.utils.runtime_config import load_runtime_env_config


def load_env_config_for_traffic(
    config_path: Path,
    traffic_path: Path,
    project_root: Path,
    preload_graphs: bool = False,
) -> dict:
    """Load env config, inject a specific traffic file, and fix all data paths."""
    return load_runtime_env_config(
        config_path,
        project_root,
        traffic_path=traffic_path,
        preload_graphs=preload_graphs,
        train_random_start=False,
    )


def baseline_policies(config: dict) -> list[tuple[str, object]]:
    """Create all baseline policies."""
    return [
        ("RSMR", RSMRPolicy.from_config(config)),
        ("ShortestDelay", ShortestDelayPolicy()),
        ("MaintainUntilConflict", MaintainUntilConflictPolicy()),
        ("GreedyConflictAware", GreedyConflictAwarePolicy()),
        ("ProactiveRule", ProactiveRuleBaseline.from_config(config)),
    ]
