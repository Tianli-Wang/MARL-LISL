"""Build sparse, per-timeslot LISL graph snapshots."""

from __future__ import annotations

import warnings
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Mapping

import numpy as np

from marl_lisl.utils.geometry import visible_segment_mask

try:
    from scipy.spatial import cKDTree
except ImportError as exc:  # pragma: no cover
    raise ImportError("scipy is required; install it with: pip install scipy") from exc


EDGE_ATTR_COLUMNS = (
    "distance_m",
    "propagation_delay_s",
    "setup_delay_s",
    "residual_lifetime_s",
    "capacity",
    "angular_rate",
)
_GRAPH_STATES: np.ndarray | None = None
_GRAPH_VALID_MASK: np.ndarray | None = None
_GRAPH_DIR: Path | None = None
_GRAPH_PARAMS: dict[str, object] = {}


def _worker_count(num_workers: int | None) -> int:
    if num_workers is None or num_workers <= 0:
        return max(1, min(4, os.cpu_count() or 1))
    return max(1, int(num_workers))


def estimate_setup_delay(
    distance_m: np.ndarray | float,
    angular_rate: np.ndarray | float,
    cfg: Mapping[str, float],
) -> np.ndarray | float:
    """Estimate simplified PAT setup delay from configurable coefficients."""
    return (
        float(cfg["base_s"])
        + float(cfg["distance_coef"]) * distance_m
        + float(cfg["angular_rate_coef"]) * angular_rate
    )


def build_snapshot(
    states: np.ndarray,
    valid: np.ndarray,
    d_max_m: float,
    earth_radius_m: float,
    speed_of_light_m_s: float,
    default_capacity: float,
    setup_delay_cfg: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Construct one sparse undirected graph using cKDTree candidate search."""
    valid_ids = np.flatnonzero(valid & np.isfinite(states).all(axis=1))
    empty_index = np.empty((2, 0), dtype=np.int64)
    empty_attr = np.empty((0, len(EDGE_ATTR_COLUMNS)), dtype=np.float64)
    if valid_ids.size < 2:
        return empty_index, empty_attr

    positions = states[valid_ids, :3]
    local_pairs = cKDTree(positions).query_pairs(r=d_max_m, output_type="ndarray")
    if local_pairs.size == 0:
        return empty_index, empty_attr
    local_pairs = np.asarray(local_pairs, dtype=np.int64).reshape(-1, 2)
    p1, p2 = positions[local_pairs[:, 0]], positions[local_pairs[:, 1]]
    visible = visible_segment_mask(p1, p2, earth_radius_m)
    local_pairs, p1, p2 = local_pairs[visible], p1[visible], p2[visible]
    if local_pairs.size == 0:
        return empty_index, empty_attr

    pairs = valid_ids[local_pairs]  # cKDTree pairs and valid_ids are ascending: i < j.
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    pairs, p1, p2 = pairs[order], p1[order], p2[order]
    distance = np.linalg.norm(p1 - p2, axis=1)
    relative_speed = np.linalg.norm(
        states[pairs[:, 0], 3:] - states[pairs[:, 1], 3:], axis=1
    )
    angular_rate = relative_speed / np.maximum(distance, 1.0)
    propagation_delay = distance / speed_of_light_m_s
    setup_delay = estimate_setup_delay(distance, angular_rate, setup_delay_cfg)
    # Columns: distance, propagation delay, setup delay, residual lifetime,
    # normalized capacity, and relative angular-rate approximation.
    edge_attr = np.column_stack((
        distance,
        propagation_delay,
        setup_delay,
        np.zeros(len(pairs), dtype=np.float64),
        np.full(len(pairs), default_capacity, dtype=np.float64),
        angular_rate,
    ))
    return pairs.T.astype(np.int64, copy=False), edge_attr


def compute_residual_lifetimes(
    graph_dir: Path, num_steps: int, num_sats: int, dt: float
) -> None:
    """Reverse-scan graph files, retaining only one timeslot dictionary in memory."""
    next_lifetime: dict[int, float] = {}
    for k in range(num_steps - 1, -1, -1):
        path = graph_dir / f"graph_{k:04d}.npz"
        with np.load(path) as graph:
            edge_index = graph["edge_index"]
            edge_attr = graph["edge_attr"]
        keys = edge_index[0] * num_sats + edge_index[1]
        lifetimes = np.fromiter(
            (next_lifetime.get(int(key), 0.0) + dt for key in keys),
            dtype=np.float64,
            count=len(keys),
        )
        edge_attr[:, 3] = lifetimes
        np.savez_compressed(path, edge_index=edge_index, edge_attr=edge_attr)
        next_lifetime = {int(key): float(value) for key, value in zip(keys, lifetimes)}
        print(f"lifetime k={k:04d}, edges={edge_index.shape[1]}")


def _init_graph_worker(
    state_path: Path,
    mask_path: Path,
    graph_dir: Path,
    params: dict[str, object],
) -> None:
    """Open shared read-only mmap arrays once in every graph worker."""
    global _GRAPH_STATES, _GRAPH_VALID_MASK, _GRAPH_DIR, _GRAPH_PARAMS
    _GRAPH_STATES = np.load(state_path, mmap_mode="r")
    _GRAPH_VALID_MASK = np.load(mask_path, mmap_mode="r")
    _GRAPH_DIR = graph_dir
    _GRAPH_PARAMS = params


def _build_and_save_snapshot(k: int) -> tuple[int, int]:
    """Build and save one independent timeslot (process-pool worker)."""
    if _GRAPH_STATES is None or _GRAPH_VALID_MASK is None or _GRAPH_DIR is None:
        raise RuntimeError("Graph worker was not initialized")
    edge_index, edge_attr = build_snapshot(
        _GRAPH_STATES[k],
        _GRAPH_VALID_MASK[k],
        float(_GRAPH_PARAMS["d_max_m"]),
        float(_GRAPH_PARAMS["earth_radius_m"]),
        float(_GRAPH_PARAMS["speed_of_light_m_s"]),
        float(_GRAPH_PARAMS["default_capacity"]),
        _GRAPH_PARAMS["setup_delay_cfg"],  # type: ignore[arg-type]
    )
    np.savez_compressed(_GRAPH_DIR / f"graph_{k:04d}.npz",
                        edge_index=edge_index, edge_attr=edge_attr)
    return k, edge_index.shape[1]


def build_graph_snapshots(
    state_dir: Path,
    graph_dir: Path,
    *,
    d_max_m: float,
    earth_radius_m: float,
    speed_of_light_m_s: float,
    dt: float,
    default_capacity: float,
    setup_delay_cfg: Mapping[str, float],
    expected_num_steps: int = 721,
    expected_num_sats: int = 6080,
    num_workers: int | None = None,
) -> int:
    """Build and save all graph snapshots, then fill residual lifetimes."""
    state_dir, graph_dir = Path(state_dir), Path(graph_dir)
    state_path, mask_path = state_dir / "sat_state_m.npy", state_dir / "valid_mask.npy"
    if not state_path.is_file() or not mask_path.is_file():
        raise FileNotFoundError("Satellite state or valid mask is missing; run step 01 first")
    if min(d_max_m, earth_radius_m, speed_of_light_m_s, dt) <= 0:
        raise ValueError("Distance, radius, light speed, and dt must be positive")

    states = np.load(state_path, mmap_mode="r")
    valid_mask = np.load(mask_path, mmap_mode="r")
    if states.ndim != 3 or states.shape[2] != 6:
        raise ValueError(f"Expected state shape (T, N, 6), got {states.shape}")
    if valid_mask.shape != states.shape[:2]:
        raise ValueError(f"State/mask shape mismatch: {states.shape} vs {valid_mask.shape}")
    t_count, n_count = valid_mask.shape
    if t_count != expected_num_steps or n_count != expected_num_sats:
        warnings.warn(
            f"Expected (T, N)=({expected_num_steps}, {expected_num_sats}), "
            f"got {(t_count, n_count)}"
        )

    graph_dir.mkdir(parents=True, exist_ok=True)
    stale = list(graph_dir.glob("graph_*.npz"))
    if stale:
        print(f"Removing {len(stale)} existing graph snapshots...")
        for path in stale:
            path.unlink()
    workers = _worker_count(num_workers)
    print(f"parallel graph workers={workers}")
    params: dict[str, object] = {
        "d_max_m": d_max_m,
        "earth_radius_m": earth_radius_m,
        "speed_of_light_m_s": speed_of_light_m_s,
        "default_capacity": default_capacity,
        "setup_delay_cfg": dict(setup_delay_cfg),
    }
    # Each timeslot writes a distinct file, so graph construction is safely parallel.
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_graph_worker,
        initargs=(state_path, mask_path, graph_dir, params),
    ) as executor:
        for k, edge_count in executor.map(_build_and_save_snapshot, range(t_count), chunksize=1):
            print(f"k={k:04d}, edges={edge_count}")

    print("Computing residual link lifetimes by dependency-ordered reverse scan...")
    compute_residual_lifetimes(graph_dir, t_count, n_count, dt)
    print(f"Saved {t_count} graph snapshots to {graph_dir}")
    return t_count
