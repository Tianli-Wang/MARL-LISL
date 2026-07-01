"""Traffic-pair array access."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class TrafficStore:
    def __init__(self, traffic_path: str | Path):
        self.traffic_path = Path(traffic_path)
        if not self.traffic_path.is_file():
            raise FileNotFoundError(
                f"Traffic file not found: {self.traffic_path}. "
                "Run scripts/preprocess/04_build_traffic.py --split normal first."
            )
        pairs = np.load(self.traffic_path)
        if pairs.ndim != 2 or pairs.shape[1] != 3:
            raise ValueError(f"Traffic pairs must have shape (num_flows, 3), got {pairs.shape}")
        if np.any(pairs[:, 0] == pairs[:, 1]):
            raise ValueError("Traffic source and destination must differ")
        self._pairs = np.asarray(pairs, dtype=np.float64)

    def get_pairs(self) -> np.ndarray:
        return self._pairs.copy()
