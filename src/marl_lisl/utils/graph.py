"""Sparse graph and path helpers; no dense adjacency matrices are created."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


def make_edge_key(u: int, v: int) -> tuple[int, int]:
    u, v = int(u), int(v)
    return (u, v) if u < v else (v, u)


def build_adjacency_from_edge_index(
    edge_index: np.ndarray, edge_attr: np.ndarray | None = None
) -> dict[int, list[tuple[int, int]]]:
    """Return undirected adjacency entries `(neighbor, edge_id)`."""
    edge_index = np.asarray(edge_index)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E), got {edge_index.shape}")
    if edge_attr is not None and len(edge_attr) != edge_index.shape[1]:
        raise ValueError("edge_index and edge_attr edge counts differ")
    adjacency: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for edge_id, (u, v) in enumerate(edge_index.T):
        u, v = int(u), int(v)
        adjacency[u].append((v, edge_id))
        adjacency[v].append((u, edge_id))
    return dict(adjacency)


def build_edge_lookup(graph: dict[str, np.ndarray]) -> dict[tuple[int, int], int]:
    return {
        make_edge_key(int(u), int(v)): edge_id
        for edge_id, (u, v) in enumerate(graph["edge_index"].T)
    }


def encode_edge_array(edge_index: np.ndarray, num_sats: int) -> np.ndarray:
    """Encode undirected edges as sortable integer keys."""
    low = np.minimum(edge_index[0], edge_index[1]).astype(np.int64, copy=False)
    high = np.maximum(edge_index[0], edge_index[1]).astype(np.int64, copy=False)
    return low * int(num_sats) + high


def encode_path_edges(node_path: Iterable[int] | None, num_sats: int) -> np.ndarray:
    """Encode a node path's undirected edges as integer keys."""
    if node_path is None:
        return np.empty(0, dtype=np.int64)
    nodes = [int(node) for node in node_path]
    if len(nodes) < 2:
        return np.empty(0, dtype=np.int64)
    num_sats = int(num_sats)
    return np.fromiter(
        (
            (u * num_sats + v) if u < v else (v * num_sats + u)
            for u, v in zip(nodes[:-1], nodes[1:])
        ),
        dtype=np.int64,
        count=len(nodes) - 1,
    )


def ensure_edge_key_index(
    graph: dict[str, np.ndarray],
    num_sats: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cached sorted edge keys and matching edge IDs for one graph."""
    num_sats = int(num_sats)
    if graph.get("_edge_key_num_sats") != num_sats:
        keys = encode_edge_array(graph["edge_index"], num_sats)
        if len(keys) > 1 and np.any(keys[1:] < keys[:-1]):
            order = np.argsort(keys, kind="mergesort")
            graph["_edge_keys_sorted"] = keys[order]
            graph["_edge_ids_sorted"] = order.astype(np.int64, copy=False)
        else:
            graph["_edge_keys_sorted"] = keys
            graph["_edge_ids_sorted"] = np.arange(len(keys), dtype=np.int64)
        # 原始图字段都是数组，但此处额外缓存整数规模标记；局部忽略这一已知的
        # 扩展字段差异，同时保留函数签名对输入数组的约束。
        graph["_edge_key_num_sats"] = num_sats  # type: ignore[assignment]
    return graph["_edge_keys_sorted"], graph["_edge_ids_sorted"]


def edge_ids_for_node_path(
    graph: dict[str, np.ndarray],
    node_path: Iterable[int] | None,
    num_sats: int,
) -> np.ndarray | None:
    """Return edge_attr indices for a node path, or None if any edge is absent."""
    encoded = encode_path_edges(node_path, num_sats)
    if not len(encoded):
        return None
    sorted_keys, sorted_edge_ids = ensure_edge_key_index(graph, num_sats)
    indices = np.searchsorted(sorted_keys, encoded)
    if np.any(indices >= len(sorted_keys)) or np.any(sorted_keys[indices] != encoded):
        return None
    return sorted_edge_ids[indices]


def edge_ids_for_edge_set(
    graph: dict[str, np.ndarray],
    edges: Iterable[tuple[int, int]],
    num_sats: int,
) -> np.ndarray:
    """Return edge_attr indices for an unordered edge iterable."""
    edge_list = list(edges)
    if not edge_list:
        return np.empty(0, dtype=np.int64)
    num_sats = int(num_sats)
    encoded = np.fromiter(
        (
            (int(u) * num_sats + int(v))
            if int(u) < int(v)
            else (int(v) * num_sats + int(u))
            for u, v in edge_list
        ),
        dtype=np.int64,
        count=len(edge_list),
    )
    sorted_keys, sorted_edge_ids = ensure_edge_key_index(graph, num_sats)
    indices = np.searchsorted(sorted_keys, encoded)
    if np.any(indices >= len(sorted_keys)) or np.any(sorted_keys[indices] != encoded):
        raise KeyError("At least one edge is absent from the graph")
    return sorted_edge_ids[indices]


def node_path_available(
    graph: dict[str, np.ndarray],
    node_path: Iterable[int] | None,
    num_sats: int,
) -> bool:
    """Return whether all path edges are present in the sparse graph."""
    return edge_ids_for_node_path(graph, node_path, num_sats) is not None


def edge_path_from_node_path(node_path: Iterable[int] | None) -> set[tuple[int, int]]:
    if node_path is None:
        return set()
    nodes = list(node_path)
    return {make_edge_key(u, v) for u, v in zip(nodes[:-1], nodes[1:])}
