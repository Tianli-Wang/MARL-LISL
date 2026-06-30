"""Build compact per-node relay capacities for mutex detection."""

from __future__ import annotations

import numpy as np


def build_node_mutex(num_sats: int, strict_node_mutex: bool = True) -> np.ndarray:
    """Return node capacities; strict mode permits one relay flow per node."""
    num_sats = int(num_sats)
    if num_sats <= 0:
        raise ValueError("num_sats must be positive")
    capacity = 1 if strict_node_mutex else num_sats
    return np.full(num_sats, capacity, dtype=np.int32)
