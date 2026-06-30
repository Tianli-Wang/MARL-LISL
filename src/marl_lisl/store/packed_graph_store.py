"""Shared-memory packed graph snapshots via memmap (zero per-step decompression).

The lazy :class:`~marl_lisl.store.graph_store.GraphStore` decompresses one ~16 MB
``graph_XXXX.npz`` per timeslot (~90 ms each). Preloading all 721 snapshots costs
~21 GB *per process*, which makes many parallel env workers infeasible.

This store packs every snapshot once into two contiguous ``.npy`` files and serves
each timeslot as a zero-copy memmap *view*. All worker processes mmap the same
files, so the ~21 GB lives in the OS page cache exactly once regardless of how many
env workers run. Per-step graph access becomes a pointer slice instead of a decode.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from marl_lisl.utils.progress import progress_iter

_META_NAME = "meta.json"
_EDGE_INDEX_NAME = "edge_index.npy"
_EDGE_ATTR_NAME = "edge_attr.npy"
_OFFSETS_NAME = "offsets.npy"


def default_pack_dir(graph_dir: str | Path) -> Path:
    return Path(graph_dir) / "_packed"


def build_graph_pack(
    graph_dir: str | Path,
    pack_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Pack all ``graph_*.npz`` snapshots into contiguous memmap-able arrays once."""
    graph_dir = Path(graph_dir)
    pack_dir = Path(pack_dir) if pack_dir is not None else default_pack_dir(graph_dir)
    meta_path = pack_dir / _META_NAME
    if meta_path.is_file() and not force:
        return pack_dir
    paths = sorted(graph_dir.glob("graph_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No graph_*.npz files found in {graph_dir}")
    pack_dir.mkdir(parents=True, exist_ok=True)

    counts: list[int] = []
    feat_dim: int | None = None
    for path in progress_iter(paths, desc="pack graphs (scan)", unit="graph"):
        with np.load(path) as data:
            edge_index = data["edge_index"]
            edge_attr = data["edge_attr"]
            counts.append(int(edge_index.shape[1]))
            feat_dim = int(edge_attr.shape[1])
    offsets = np.zeros(len(paths) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    total_edges = int(offsets[-1])

    edge_index_mm = np.lib.format.open_memmap(
        pack_dir / _EDGE_INDEX_NAME, mode="w+", dtype=np.int64, shape=(2, total_edges)
    )
    edge_attr_mm = np.lib.format.open_memmap(
        pack_dir / _EDGE_ATTR_NAME, mode="w+", dtype=np.float32, shape=(total_edges, feat_dim)
    )
    ks: list[int] = []
    for index, path in enumerate(
        progress_iter(paths, desc="pack graphs (write)", unit="graph")
    ):
        ks.append(int(path.stem.split("_")[-1]))
        start, end = int(offsets[index]), int(offsets[index + 1])
        with np.load(path) as data:
            edge_index_mm[:, start:end] = np.asarray(data["edge_index"], dtype=np.int64)
            edge_attr_mm[start:end] = np.asarray(data["edge_attr"], dtype=np.float32)
    edge_index_mm.flush()
    edge_attr_mm.flush()
    del edge_index_mm, edge_attr_mm
    np.save(pack_dir / _OFFSETS_NAME, offsets)
    with (pack_dir / _META_NAME).open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "num_graphs": len(paths),
                "feat_dim": int(feat_dim),
                "ks": ks,
                "total_edges": total_edges,
            },
            handle,
        )
    print(
        f"packed {len(paths)} graph snapshots -> {pack_dir} "
        f"({(2 * total_edges * 8 + total_edges * feat_dim * 4) / 1024 ** 3:.2f} GiB shared memmap)"
    )
    return pack_dir


class PackedGraphStore:
    """Memmap-backed, API-compatible drop-in for :class:`GraphStore`.

    Returns a per-timeslot dict of zero-copy memmap views, cached so the sorted
    edge-key index added by :func:`marl_lisl.utils.graph.ensure_edge_key_index`
    persists while the timeslot stays in the LRU.
    """

    def __init__(
        self,
        graph_dir: str | Path,
        pack_dir: str | Path | None = None,
        cache_size: int = 64,
        build_if_missing: bool = True,
    ):
        self.graph_dir = Path(graph_dir)
        self.pack_dir = Path(pack_dir) if pack_dir is not None else default_pack_dir(graph_dir)
        meta_path = self.pack_dir / _META_NAME
        if not meta_path.is_file():
            if build_if_missing:
                build_graph_pack(self.graph_dir, self.pack_dir)
            else:
                raise FileNotFoundError(
                    f"Graph pack not found: {meta_path}. Build it with build_graph_pack()."
                )
        with meta_path.open(encoding="utf-8") as handle:
            meta = json.load(handle)
        self.feat_dim = int(meta["feat_dim"])
        self.ks: list[int] = [int(k) for k in meta["ks"]]
        self._k_to_index = {k: i for i, k in enumerate(self.ks)}
        self.offsets = np.load(self.pack_dir / _OFFSETS_NAME)
        # Read-only memmaps; pages are shared across processes via the OS page cache.
        self.edge_index = np.load(self.pack_dir / _EDGE_INDEX_NAME, mmap_mode="r")
        self.edge_attr = np.load(self.pack_dir / _EDGE_ATTR_NAME, mmap_mode="r")
        self.cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()

    def get_graph(self, k: int) -> dict[str, np.ndarray]:
        k = int(k)
        if k < 0:
            raise IndexError(f"Graph timeslot must be non-negative, got {k}")
        cached = self._cache.get(k)
        if cached is not None:
            self._cache.move_to_end(k)
            return cached
        index = self._k_to_index.get(k)
        if index is None:
            raise FileNotFoundError(
                f"Graph snapshot not found for timeslot {k} in pack {self.pack_dir}."
            )
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        graph = {
            "edge_index": self.edge_index[:, start:end],
            "edge_attr": self.edge_attr[start:end],
        }
        self._cache[k] = graph
        self._cache.move_to_end(k)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return graph

    def clear_cache(self) -> None:
        self._cache.clear()
