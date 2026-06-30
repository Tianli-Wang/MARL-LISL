"""Candidate-path observations for per-flow agents."""

from __future__ import annotations

import numpy as np

from marl_lisl.utils.graph import build_edge_lookup, edge_path_from_node_path


class ObservationBuilder:
    # T_prop, T_setup, R_min, N_new, hops, feasible, A_mutex, B_avoid
    obs_dim = 8

    def __init__(self, num_candidates: int):
        self.num_candidates = int(num_candidates)
        self._graph_token: int | None = None
        self._edge_lookup: dict[tuple[int, int], int] = {}

    def _lookup(self, graph: dict) -> dict[tuple[int, int], int]:
        token = id(graph["edge_index"])
        if token != self._graph_token:
            self._edge_lookup = build_edge_lookup(graph)
            self._graph_token = token
        return self._edge_lookup

    def path_features(
        self, graph: dict, current_path: list[int] | None, path: list[int] | None
    ) -> np.ndarray:
        result = np.zeros(self.obs_dim, dtype=np.float32)
        if not path or len(path) < 2:
            return result
        lookup = self._lookup(graph)
        path_edges = edge_path_from_node_path(path)
        if not path_edges or any(edge not in lookup for edge in path_edges):
            return result
        old_edges = edge_path_from_node_path(current_path)
        edge_ids = np.asarray([lookup[edge] for edge in path_edges], dtype=np.int64)
        new_edges = path_edges - old_edges
        new_ids = [lookup[edge] for edge in new_edges]
        attrs = graph["edge_attr"]
        result[:] = (
            float(attrs[edge_ids, 1].sum()),
            float(attrs[new_ids, 2].max()) if new_ids else 0.0,
            float(attrs[edge_ids, 3].min()),
            float(len(new_edges)),
            float(len(path) - 1),
            1.0,
            0.0,
            0.0,
        )
        return result

    def build_flow_obs(
        self,
        graph: dict,
        current_path: list[int] | None,
        candidate_paths: list[list[int]],
        future_mutex_detector=None,
        all_paths: list | None = None,
        flow_id: int | None = None,
        k: int | None = None,
        future_mutex_keep: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        obs = np.zeros((self.num_candidates + 1, self.obs_dim), dtype=np.float32)
        mask = np.zeros(self.num_candidates + 1, dtype=np.float32)
        if current_path is not None:
            obs[0] = self.path_features(graph, current_path, current_path)
            mask[0] = obs[0, 5]
        mutex_enabled = (
            future_mutex_detector is not None
            and all_paths is not None
            and flow_id is not None
            and k is not None
        )
        if mutex_enabled and future_mutex_keep is None:
            future_mutex_keep, _ = future_mutex_detector.compute_future_mutex(all_paths, k)
        keep_mutex = float(future_mutex_keep or 0.0)
        if mask[0] > 0:
            obs[0, 6] = keep_mutex
            obs[0, 7] = 0.0
        for index, path in enumerate(candidate_paths[: self.num_candidates], start=1):
            obs[index] = self.path_features(graph, current_path, path)
            mask[index] = obs[index, 5]
            if mutex_enabled and mask[index] > 0:
                candidate_mutex, _ = future_mutex_detector.compute_candidate_mutex(
                    all_paths, flow_id, path, k
                )
                obs[index, 6] = candidate_mutex
                obs[index, 7] = keep_mutex - candidate_mutex
        return obs, mask
