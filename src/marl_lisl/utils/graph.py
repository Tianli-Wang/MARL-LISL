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


def edge_path_from_node_path(node_path: Iterable[int] | None) -> set[tuple[int, int]]:
    if node_path is None:
        return set()
    nodes = list(node_path)
    return {make_edge_key(u, v) for u, v in zip(nodes[:-1], nodes[1:])}
