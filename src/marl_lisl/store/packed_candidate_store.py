"""Memmap-backed candidate path lookup across all timeslots."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from marl_lisl.utils.progress import progress_iter

_META_NAME = "meta.json"
_NODES_NAME = "nodes.npy"
_OFFSETS_NAME = "offsets.npy"


def default_candidate_pack_dir(candidate_dir: str | Path) -> Path:
    return Path(candidate_dir) / "_packed"


def build_candidate_pack(
    candidate_dir: str | Path,
    pack_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Pack per-slot ``cand_XXXX.npz`` files into shared memmap arrays."""
    candidate_dir = Path(candidate_dir)
    pack_dir = Path(pack_dir) if pack_dir is not None else default_candidate_pack_dir(candidate_dir)
    meta_path = pack_dir / _META_NAME
    if meta_path.is_file() and not force:
        return pack_dir
    paths = sorted(candidate_dir.glob("cand_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No cand_*.npz files found in {candidate_dir}")
    pack_dir.mkdir(parents=True, exist_ok=True)

    ks: list[int] = []
    node_counts: list[int] = []
    num_flows: int | None = None
    num_candidates: int | None = None
    for path in progress_iter(paths, desc="pack candidates (scan)", unit="slot"):
        with np.load(path) as data:
            nodes = data["nodes"]
            offsets = data["offsets"]
            if offsets.ndim != 2:
                raise ValueError(f"{path}: offsets must be 2-D, got {offsets.shape}")
            if num_flows is None:
                num_flows = int(offsets.shape[0])
                num_candidates = int(offsets.shape[1] - 1)
            elif offsets.shape != (num_flows, num_candidates + 1):
                raise ValueError(
                    f"{path}: offsets shape {offsets.shape} does not match "
                    f"({num_flows}, {num_candidates + 1})"
                )
            ks.append(int(path.stem.split("_")[-1]))
            node_counts.append(int(len(nodes)))

    total_nodes = int(sum(node_counts))
    nodes_mm = np.lib.format.open_memmap(
        pack_dir / _NODES_NAME, mode="w+", dtype=np.int64, shape=(total_nodes,)
    )
    offsets_mm = np.lib.format.open_memmap(
        pack_dir / _OFFSETS_NAME,
        mode="w+",
        dtype=np.int64,
        shape=(len(paths), int(num_flows), int(num_candidates) + 1),
    )

    cursor = 0
    for slot_id, path in enumerate(
        progress_iter(paths, desc="pack candidates (write)", unit="slot")
    ):
        with np.load(path) as data:
            nodes = np.asarray(data["nodes"], dtype=np.int64)
            offsets = np.asarray(data["offsets"], dtype=np.int64)
        start = cursor
        cursor += int(len(nodes))
        nodes_mm[start:cursor] = nodes
        offsets_mm[slot_id] = offsets + start
    nodes_mm.flush()
    offsets_mm.flush()
    del nodes_mm, offsets_mm

    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "num_steps": len(paths),
                "num_flows": int(num_flows),
                "num_candidates": int(num_candidates),
                "ks": ks,
                "total_nodes": total_nodes,
                "format": "packed-candidates-v1",
            },
            handle,
            indent=2,
        )
    print(
        f"packed {len(paths)} candidate files -> {pack_dir} "
        f"({total_nodes} path nodes)"
    )
    return pack_dir


class PackedCandidateStore:
    """API-compatible candidate store backed by two memmap arrays."""

    def __init__(
        self,
        candidate_dir: str | Path,
        num_flows: int,
        num_candidates: int,
        pack_dir: str | Path | None = None,
        cache_size: int = 64,
        build_if_missing: bool = True,
        expected_num_steps: int | None = None,
    ):
        self.candidate_dir = Path(candidate_dir)
        self.pack_dir = (
            Path(pack_dir) if pack_dir is not None else default_candidate_pack_dir(candidate_dir)
        )
        meta_path = self.pack_dir / _META_NAME
        if not meta_path.is_file():
            if build_if_missing:
                build_candidate_pack(self.candidate_dir, self.pack_dir)
            else:
                raise FileNotFoundError(
                    f"Candidate pack not found: {meta_path}. Build it with build_candidate_pack()."
                )
        with meta_path.open(encoding="utf-8") as handle:
            meta = json.load(handle)
        self.num_flows = int(num_flows)
        self.num_candidates = int(num_candidates)
        if int(meta["num_flows"]) != self.num_flows:
            raise ValueError(
                f"Candidate pack has {meta['num_flows']} flows, expected {self.num_flows}"
            )
        if int(meta["num_candidates"]) < self.num_candidates:
            raise ValueError(
                f"Candidate pack has {meta['num_candidates']} candidates, "
                f"expected {self.num_candidates}"
            )
        self.ks = [int(k) for k in meta["ks"]]
        if expected_num_steps is not None:
            expected = set(range(int(expected_num_steps)))
            present = set(self.ks)
            missing = sorted(expected - present)
            if missing:
                preview = ", ".join(str(k) for k in missing[:8])
                suffix = "..." if len(missing) > 8 else ""
                raise ValueError(
                    f"Candidate pack {self.pack_dir} is incomplete: missing "
                    f"{len(missing)} timeslots ({preview}{suffix}). "
                    "Rerun scripts/preprocess/06_build_candidates.py."
                )
        self._k_to_index = {k: i for i, k in enumerate(self.ks)}
        self.nodes = np.load(self.pack_dir / _NODES_NAME, mmap_mode="r")
        self.offsets = np.load(self.pack_dir / _OFFSETS_NAME, mmap_mode="r")
        self.cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[int, list[list[list[int]]]] = OrderedDict()

    def get_candidates(self, k: int) -> list[list[list[int]]]:
        k = int(k)
        if k in self._cache:
            self._cache.move_to_end(k)
            return self._clone(self._cache[k])
        slot_id = self._k_to_index.get(k)
        if slot_id is None:
            raise FileNotFoundError(
                f"Candidate timeslot {k} not found in pack {self.pack_dir}"
            )
        offsets = self.offsets[slot_id]
        candidates: list[list[list[int]]] = []
        for flow_id in range(self.num_flows):
            flow_paths: list[list[int]] = []
            for candidate_id in range(self.num_candidates):
                start = int(offsets[flow_id, candidate_id])
                end = int(offsets[flow_id, candidate_id + 1])
                flow_paths.append(self.nodes[start:end].astype(int).tolist())
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
        self._cache.clear()
