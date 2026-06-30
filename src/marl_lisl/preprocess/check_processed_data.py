"""Validation routines for satellite-state arrays and graph snapshots."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from marl_lisl.utils.progress import progress_iter


def _warn(message: str) -> None:
    warnings.warn(message, stacklevel=2)


def _print_stats(label: str, values: np.ndarray) -> None:
    if values.size:
        print(f"  {label}: min={values.min():.9g}, max={values.max():.9g}, "
              f"mean={values.mean():.9g}")
    else:
        print(f"  {label}: no edges")


def check_graph(path: Path, num_sats: int) -> None:
    """Print statistics and non-fatally validate one graph file."""
    try:
        with np.load(path) as graph:
            if "edge_index" not in graph or "edge_attr" not in graph:
                _warn(f"{path.name}: missing edge_index or edge_attr")
                return
            edge_index, edge_attr = graph["edge_index"], graph["edge_attr"]
    except Exception as exc:
        _warn(f"{path.name}: cannot load graph: {exc}")
        return

    print(f"\n{path.name}")
    print(f"  edge_index.shape={edge_index.shape}")
    print(f"  edge_attr.shape={edge_attr.shape}")
    edge_count = edge_index.shape[1] if edge_index.ndim == 2 else 0
    print(f"  edges={edge_count}")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        _warn(f"{path.name}: edge_index must have shape (2, E)")
        return
    if edge_attr.ndim != 2 or edge_attr.shape[1] != 6:
        _warn(f"{path.name}: edge_attr must have shape (E, 6)")
        return
    if edge_attr.shape[0] != edge_count:
        _warn(f"{path.name}: edge counts differ")
        return

    _print_stats("distance_m", edge_attr[:, 0])
    _print_stats("propagation_delay_s", edge_attr[:, 1])
    _print_stats("residual_lifetime_s", edge_attr[:, 3])
    if np.isnan(edge_attr).any() or not np.isfinite(edge_attr).all():
        _warn(f"{path.name}: edge_attr contains NaN/non-finite values")
    for column, name in ((0, "distance"), (1, "propagation delay"), (3, "lifetime")):
        if np.any(edge_attr[:, column] < 0):
            _warn(f"{path.name}: negative {name} detected")
    if not np.issubdtype(edge_index.dtype, np.integer):
        _warn(f"{path.name}: edge_index is not integer")
    if edge_count and (edge_index.min() < 0 or edge_index.max() >= num_sats):
        _warn(f"{path.name}: edge_index out of range [0, {num_sats})")
    if np.any(edge_index[0] == edge_index[1]):
        _warn(f"{path.name}: self-loop detected")
    if np.any(edge_index[0] >= edge_index[1]):
        _warn(f"{path.name}: non-canonical edge detected (expected i < j)")
    if edge_count and len(np.unique(edge_index.T, axis=0)) != edge_count:
        _warn(f"{path.name}: duplicate edge detected")


def check_processed_data(
    state_dir: Path,
    graph_dir: Path,
    samples: int = 5,
    seed: int = 0,
) -> None:
    """Validate state files and a deterministic random sample of snapshots."""
    state_dir, graph_dir = Path(state_dir), Path(graph_dir)
    state_path, mask_path = state_dir / "sat_state_m.npy", state_dir / "valid_mask.npy"
    missing = [path for path in (state_path, mask_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing critical files: {', '.join(map(str, missing))}")
    if not graph_dir.is_dir():
        raise FileNotFoundError(f"Missing critical graph directory: {graph_dir}")

    states = np.load(state_path, mmap_mode="r")
    valid_mask = np.load(mask_path, mmap_mode="r")
    print(f"sat_state_m.shape={states.shape}")
    print(f"valid_mask.shape={valid_mask.shape}")
    if states.ndim != 3 or states.shape[-1] != 6:
        _warn("sat_state_m must have shape (T, N, 6)")
    if valid_mask.ndim != 2 or valid_mask.shape != states.shape[:2]:
        _warn("valid_mask shape does not match state")
    if valid_mask.dtype != np.bool_:
        _warn(f"valid_mask dtype should be bool, got {valid_mask.dtype}")

    if states.ndim == 3 and valid_mask.shape == states.shape[:2]:
        valid_nan = invalid_value = False
        slots = progress_iter(
            range(states.shape[0]),
            total=states.shape[0],
            desc="03 串行检查状态数组",
            unit="slot",
        )
        for k in slots:
            mask = valid_mask[k]
            valid_nan |= bool(np.isnan(states[k, mask]).any())
            invalid_value |= bool(np.isfinite(states[k, ~mask]).any())
        if valid_nan:
            _warn("valid satellite states contain NaN")
        if invalid_value:
            _warn("invalid satellite states are not entirely NaN")

    graph_paths = sorted(graph_dir.glob("graph_*.npz"))
    if not graph_paths:
        raise FileNotFoundError(f"No graph_*.npz files found in {graph_dir}")
    if states.ndim and len(graph_paths) != states.shape[0]:
        _warn(f"Expected {states.shape[0]} graph files, found {len(graph_paths)}")
    sample_count = min(max(int(samples), 0), len(graph_paths))
    indices = sorted(np.random.default_rng(seed).choice(
        len(graph_paths), size=sample_count, replace=False
    ).tolist())
    print(f"graph files={len(graph_paths)}, sampled={sample_count}, indices={indices}")
    num_sats = states.shape[1] if states.ndim >= 2 else 0
    for index in progress_iter(indices, desc="03 检查抽样图快照", unit="graph"):
        check_graph(graph_paths[index], num_sats)
