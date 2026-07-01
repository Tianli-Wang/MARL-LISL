"""Precompute candidate paths for every graph snapshot and traffic split."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Literal

import numpy as np

from marl_lisl.envs.path_generator import PathGenerator
from marl_lisl.store.graph_store import GraphStore
from marl_lisl.store.packed_candidate_store import build_candidate_pack
from marl_lisl.store.packed_graph_store import PackedGraphStore, build_graph_pack
from marl_lisl.utils.progress import progress_iter

_CANDIDATE_GRAPH_DIR: Path | None = None
_CANDIDATE_GRAPH_BACKEND: str = "lazy"
_CANDIDATE_GRAPH_PACK_DIR: Path | None = None
_CANDIDATE_TRAFFIC: np.ndarray | None = None
_CANDIDATE_NUM_CANDIDATES: int = 0
_CANDIDATE_PATH_WEIGHT: dict[str, float] = {}
_CANDIDATE_OUTPUT_DIR: Path | None = None


def _worker_count(num_workers: int | None) -> int:
    if num_workers is None or num_workers <= 0:
        return max(1, min(128, os.cpu_count() or 1))
    return max(1, int(num_workers))


def pack_candidate_paths(
    candidates: list[list[list[int]]],
    num_flows: int,
    num_candidates: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack variable-length paths into flat nodes plus offsets."""
    offsets = np.zeros((num_flows, num_candidates + 1), dtype=np.int64)
    flat_nodes: list[int] = []
    for flow_id in range(num_flows):
        flow_paths = candidates[flow_id] if flow_id < len(candidates) else []
        for candidate_id in range(num_candidates):
            offsets[flow_id, candidate_id] = len(flat_nodes)
            path = flow_paths[candidate_id] if candidate_id < len(flow_paths) else []
            flat_nodes.extend(int(node) for node in path)
        offsets[flow_id, num_candidates] = len(flat_nodes)
    return np.asarray(flat_nodes, dtype=np.int64), offsets


def _init_candidate_worker(
    graph_dir: Path,
    graph_backend: str,
    graph_pack_dir: Path | None,
    traffic_pairs: np.ndarray,
    num_candidates: int,
    path_weight: dict,
    output_dir: Path,
) -> None:
    """Initialize read-only inputs once in each process-pool worker."""
    global _CANDIDATE_GRAPH_DIR, _CANDIDATE_GRAPH_BACKEND, _CANDIDATE_GRAPH_PACK_DIR
    global _CANDIDATE_TRAFFIC, _CANDIDATE_NUM_CANDIDATES
    global _CANDIDATE_PATH_WEIGHT, _CANDIDATE_OUTPUT_DIR
    _CANDIDATE_GRAPH_DIR = Path(graph_dir)
    _CANDIDATE_GRAPH_BACKEND = str(graph_backend)
    _CANDIDATE_GRAPH_PACK_DIR = None if graph_pack_dir is None else Path(graph_pack_dir)
    _CANDIDATE_TRAFFIC = np.asarray(traffic_pairs, dtype=np.float64)
    _CANDIDATE_NUM_CANDIDATES = int(num_candidates)
    _CANDIDATE_PATH_WEIGHT = dict(path_weight)
    _CANDIDATE_OUTPUT_DIR = Path(output_dir)


def _build_one_candidate_file(k: int) -> tuple[int, int, int]:
    """Build candidates for one timeslot and save one cand_XXXX.npz."""
    if (
        _CANDIDATE_GRAPH_DIR is None
        or _CANDIDATE_TRAFFIC is None
        or _CANDIDATE_OUTPUT_DIR is None
    ):
        raise RuntimeError("Candidate worker was not initialized")
    if _CANDIDATE_GRAPH_BACKEND == "packed":
        graph_store = PackedGraphStore(
            _CANDIDATE_GRAPH_DIR,
            pack_dir=_CANDIDATE_GRAPH_PACK_DIR,
            cache_size=1,
            build_if_missing=False,
        )
    else:
        graph_store = GraphStore(_CANDIDATE_GRAPH_DIR, cache_size=1)
    graph = graph_store.get_graph(k)
    generator = PathGenerator(_CANDIDATE_NUM_CANDIDATES, _CANDIDATE_PATH_WEIGHT)
    generator.prepare_graph(graph)

    candidates: list[list[list[int]]] = []
    non_empty = 0
    total_paths = 0
    for source, dest, _demand in _CANDIDATE_TRAFFIC:
        paths = generator.generate(graph, int(source), int(dest))
        non_empty += int(bool(paths))
        total_paths += len(paths)
        candidates.append(paths)
    nodes, offsets = pack_candidate_paths(
        candidates, len(_CANDIDATE_TRAFFIC), _CANDIDATE_NUM_CANDIDATES
    )
    np.savez_compressed(
        _CANDIDATE_OUTPUT_DIR / f"cand_{k:04d}.npz",
        nodes=nodes,
        offsets=offsets,
    )
    return int(k), int(total_paths), int(non_empty)


CandidateSplit = Literal["train", "eval", "stress"]


def _traffic_path_for_split(config: dict, split: CandidateSplit) -> Path:
    key = {
        "train": "traffic_train_path",
        "eval": "traffic_eval_path",
        "stress": "traffic_stress_path",
    }[split]
    return Path(config[key])


def _candidate_dir_for_split(
    config: dict,
    project_root: Path,
    split: CandidateSplit,
) -> Path:
    candidates_cfg = config.get("candidates", {})
    key = {
        "train": "train_dir",
        "eval": "eval_dir",
        "stress": "stress_dir",
    }[split]
    return project_root / candidates_cfg[key]


def build_candidates_for_split(
    config: dict,
    project_root: Path,
    split: CandidateSplit,
    num_workers: int | None = None,
) -> Path:
    """Precompute and save candidates for one traffic split."""
    project_root = Path(project_root)
    graph_dir = project_root / config["graph_dir"]
    graph_backend = str(config.get("graph_backend", "lazy")).lower()
    graph_pack_dir = (
        project_root / config["graph_pack_dir"]
        if config.get("graph_pack_dir") is not None
        else None
    )
    if graph_backend == "packed":
        # packed graph 必须在创建进程池之前由主进程一次性准备。worker 中保持
        # build_if_missing=False，能够避免大量子进程同时检测并写入同一组 memmap。
        effective_pack_dir = (
            graph_pack_dir if graph_pack_dir is not None else graph_dir / "_packed"
        )
        meta_path = effective_pack_dir / "meta.json"
        if not meta_path.is_file():
            if bool(config.get("graph_pack_build_if_missing", True)):
                print(f"graph pack missing; building once in parent: {effective_pack_dir}")
                build_graph_pack(graph_dir, pack_dir=effective_pack_dir)
            else:
                raise FileNotFoundError(
                    f"Graph pack not found: {meta_path}. Run "
                    "scripts/preprocess/06_pack_data.py --target graphs first."
                )
    traffic_path = project_root / _traffic_path_for_split(config, split)
    if not traffic_path.is_file():
        raise FileNotFoundError(
            f"Traffic file not found: {traffic_path}. "
            "Run scripts/preprocess/04_build_traffic.py first."
        )
    traffic_pairs = np.load(traffic_path)
    if traffic_pairs.ndim != 2 or traffic_pairs.shape[1] != 3:
        raise ValueError(f"Traffic pairs must have shape (F, 3), got {traffic_pairs.shape}")
    expected_flows = int(config["num_flows"]) if split != "stress" else len(traffic_pairs)
    if len(traffic_pairs) != expected_flows:
        raise ValueError(f"Expected {expected_flows} flows but {traffic_path} has {len(traffic_pairs)} pairs")

    output_dir = _candidate_dir_for_split(config, project_root, split)
    output_dir.mkdir(parents=True, exist_ok=True)
    stale = list(output_dir.glob("cand_*.npz"))
    if stale:
        for path in progress_iter(stale, desc=f"05 删除旧 {split} candidates", unit="file"):
            path.unlink()

    num_steps = int(config["num_steps"])
    workers = _worker_count(num_workers if num_workers is not None else config.get("parallel_workers"))
    print(
        f"building {split} candidates: steps={num_steps}, flows={len(traffic_pairs)}, "
        f"K={config['num_candidates']}, workers={workers}"
    )
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_candidate_worker,
        initargs=(
            graph_dir,
            graph_backend,
            graph_pack_dir,
            traffic_pairs,
            int(config["num_candidates"]),
            dict(config["path_weight"]),
            output_dir,
        ),
    ) as executor:
        results = list(
            progress_iter(
                executor.map(_build_one_candidate_file, range(num_steps), chunksize=1),
                total=num_steps,
                desc=f"05 预计算 {split} candidates",
                unit="slot",
            )
        )
    total_paths = int(sum(item[1] for item in results))
    non_empty = int(sum(item[2] for item in results))
    metadata = {
        "split": split,
        "num_steps": num_steps,
        "num_flows": int(len(traffic_pairs)),
        "num_candidates": int(config["num_candidates"]),
        "traffic_path": str(traffic_path),
        "graph_dir": str(graph_dir),
        "graph_backend": graph_backend,
        "graph_pack_dir": None if graph_pack_dir is None else str(graph_pack_dir),
        "total_paths": total_paths,
        "non_empty_flow_slots": non_empty,
        "format": "nodes+offsets",
    }
    with (output_dir / "candidate_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    if str(config.get("candidates", {}).get("backend", "npz")).lower() == "packed":
        build_candidate_pack(output_dir, force=True)
    print(f"saved {split} candidates to {output_dir}")
    print(f"total_paths={total_paths}, non_empty_flow_slots={non_empty}")
    return output_dir


def build_candidates(
    config: dict,
    project_root: Path,
    split: str = "both",
    num_workers: int | None = None,
) -> list[Path]:
    """Precompute candidate paths for train, eval, stress, or all splits."""
    if split not in {"train", "eval", "stress", "both", "all"}:
        raise ValueError("split must be one of: train, eval, stress, both, all")
    splits: list[CandidateSplit]
    if split == "both":
        splits = ["train", "eval"]
    elif split == "all":
        splits = ["train", "eval", "stress"]
    else:
        splits = [split]  # type: ignore[list-item]
    return [
        build_candidates_for_split(config, project_root, item, num_workers)
        for item in splits
    ]
