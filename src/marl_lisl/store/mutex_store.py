"""Compact node-capacity storage for relay mutex constraints."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class MutexStore:
    def __init__(self, node_mutex_path: str | Path):
        self.node_mutex_path = Path(node_mutex_path)
        if not self.node_mutex_path.is_file():
            raise FileNotFoundError(
                f"Node mutex file not found: {self.node_mutex_path}. "
                "Run scripts/preprocess/05_build_mutex.py first."
            )
        capacity = np.load(self.node_mutex_path)
        if capacity.ndim != 1:
            raise ValueError(f"node_mutex must be one-dimensional, got {capacity.shape}")
        if not np.issubdtype(capacity.dtype, np.integer):
            raise ValueError(f"node_mutex must contain integers, got {capacity.dtype}")
        if np.any(capacity < 1):
            raise ValueError("Every node capacity must be at least 1")
        self._node_capacity = np.asarray(capacity, dtype=np.int32)

    def get_node_capacity(self) -> np.ndarray:
        return self._node_capacity.copy()

    def get_capacity(self, node_id: int) -> int:
        node_id = int(node_id)
        if not 0 <= node_id < len(self._node_capacity):
            raise IndexError(f"node_id out of range: {node_id}")
        return int(self._node_capacity[node_id])
