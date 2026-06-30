"""Lazy graph snapshot access with a small LRU cache."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np

from marl_lisl.utils.progress import progress_iter


class GraphStore:
    def __init__(self, graph_dir: str | Path, cache_size: int = 3, preload: bool = False):
        self.graph_dir = Path(graph_dir)
        self.cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self.preload = bool(preload)
        if not self.graph_dir.is_dir():
            raise FileNotFoundError(
                f"Graph directory not found: {self.graph_dir}. "
                "Run the stage-1 graph preprocessing first."
            )
        if self.preload:
            self.preload_all()

    def _load_graph(self, k: int) -> dict[str, np.ndarray]:
        path = self.graph_dir / f"graph_{k:04d}.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"Graph snapshot not found: {path}. "
                "Run scripts/preprocess/02_build_graph_snapshots.py first."
            )
        with np.load(path) as data:
            if "edge_index" not in data or "edge_attr" not in data:
                raise ValueError(f"{path} must contain edge_index and edge_attr")
            edge_index = np.asarray(data["edge_index"], dtype=np.int64)
            edge_attr = np.asarray(data["edge_attr"], dtype=np.float32)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"Invalid edge_index shape in {path}: {edge_index.shape}")
        if edge_attr.ndim != 2 or edge_attr.shape != (edge_index.shape[1], 6):
            raise ValueError(f"Invalid edge_attr shape in {path}: {edge_attr.shape}")
        return {"edge_index": edge_index, "edge_attr": edge_attr}

    def preload_all(self) -> None:
        """Load all graph snapshots into memory; useful on large-memory servers."""
        paths = sorted(self.graph_dir.glob("graph_*.npz"))
        if not paths:
            raise FileNotFoundError(f"No graph_*.npz files found in {self.graph_dir}")
        self.cache_size = max(self.cache_size, len(paths))
        for path in progress_iter(paths, desc="preload graph snapshots", unit="graph"):
            k = int(path.stem.split("_")[-1])
            self._cache[k] = self._load_graph(k)
        self._cache = OrderedDict(sorted(self._cache.items()))
        raw_bytes = sum(
            graph["edge_index"].nbytes + graph["edge_attr"].nbytes
            for graph in self._cache.values()
        )
        print(
            f"preloaded {len(self._cache)} graph snapshots "
            f"({raw_bytes / 1024 ** 3:.2f} GiB raw arrays)"
        )

    def get_graph(self, k: int) -> dict[str, np.ndarray]:
        k = int(k)
        if k < 0:
            raise IndexError(f"Graph timeslot must be non-negative, got {k}")
        if k in self._cache:
            self._cache.move_to_end(k)
            return self._cache[k]

        graph = self._load_graph(k)
        self._cache[k] = graph
        self._cache.move_to_end(k)
        while not self.preload and len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return graph

    def clear_cache(self) -> None:
        self._cache.clear()
