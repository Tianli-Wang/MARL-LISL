"""Lazy access to precomputed candidate paths."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np


class CandidateStore:
    """Load `cand_XXXX.npz` files that store variable-length node paths."""

    def __init__(
        self,
        candidate_dir: str | Path,
        num_flows: int,
        num_candidates: int,
        cache_size: int = 3,
    ):
        self.candidate_dir = Path(candidate_dir)
        self.num_flows = int(num_flows)
        self.num_candidates = int(num_candidates)
        self.cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[int, list[list[list[int]]]] = OrderedDict()
        if not self.candidate_dir.is_dir():
            raise FileNotFoundError(
                f"Candidate directory not found: {self.candidate_dir}. "
                "Run scripts/preprocess/05_build_candidates.py first."
            )

    def get_candidates(self, k: int) -> list[list[list[int]]]:
        """Return precomputed candidates as [flow][candidate][node]."""
        k = int(k)
        if k < 0:
            raise IndexError(f"Candidate timeslot must be non-negative, got {k}")
        if k in self._cache:
            self._cache.move_to_end(k)
            return self._clone(self._cache[k])

        path = self.candidate_dir / f"cand_{k:04d}.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"Candidate file not found: {path}. "
                "Run scripts/preprocess/05_build_candidates.py first."
            )
        with np.load(path) as data:
            if "nodes" not in data or "offsets" not in data:
                raise ValueError(f"{path} must contain nodes and offsets")
            nodes = np.asarray(data["nodes"], dtype=np.int64)
            offsets = np.asarray(data["offsets"], dtype=np.int64)
        if offsets.ndim != 2 or offsets.shape[0] != self.num_flows:
            raise ValueError(
                f"{path}: offsets shape {offsets.shape} has incompatible flow dimension"
            )
        if offsets.shape[1] < self.num_candidates + 1:
            raise ValueError(
                f"{path}: stores only {offsets.shape[1] - 1} candidates, "
                f"but environment requests {self.num_candidates}"
            )
        if offsets.size and (offsets.min() < 0 or offsets.max() > len(nodes)):
            raise ValueError(f"{path}: offsets out of node-array bounds")
        candidates: list[list[list[int]]] = []
        for flow_id in range(self.num_flows):
            flow_paths: list[list[int]] = []
            for candidate_id in range(self.num_candidates):
                start = int(offsets[flow_id, candidate_id])
                end = int(offsets[flow_id, candidate_id + 1])
                flow_paths.append(nodes[start:end].astype(int).tolist())
            candidates.append(flow_paths)

        self._cache[k] = candidates
        self._cache.move_to_end(k)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return self._clone(candidates)

    @staticmethod
    def _clone(candidates: list[list[list[int]]]) -> list[list[list[int]]]:
        return [[list(path) for path in flow_paths] for flow_paths in candidates]

    def clear_cache(self) -> None:
        """Clear loaded candidate files."""
        self._cache.clear()
