"""Candidate-path observations for per-flow agents."""

from __future__ import annotations

import numpy as np

from marl_lisl.utils.graph import (
    edge_ids_for_edge_set,
    edge_ids_for_node_path,
    edge_path_from_node_path,
)


class ObservationBuilder:
    # T_prop, T_setup, R_min, N_new, hops, feasible, A_mutex, B_avoid
    obs_dim = 8

    def __init__(self, num_candidates: int, num_sats: int):
        self.num_candidates = int(num_candidates)
        self.num_sats = int(num_sats)

    def path_features(
        self, graph: dict, current_path: list[int] | None, path: list[int] | None
    ) -> np.ndarray:
        result = np.zeros(self.obs_dim, dtype=np.float32)
        if not path or len(path) < 2:
            return result
        edge_ids = edge_ids_for_node_path(graph, path, self.num_sats)
        if edge_ids is None:
            return result
        path_edges = edge_path_from_node_path(path)
        old_edges = edge_path_from_node_path(current_path)
        new_edges = path_edges - old_edges
        new_ids = edge_ids_for_edge_set(graph, new_edges, self.num_sats) if new_edges else None
        attrs = graph["edge_attr"]
        result[:] = (
            float(attrs[edge_ids, 1].sum()),
            float(attrs[new_ids, 2].max()) if new_ids is not None and len(new_ids) else 0.0,
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
        candidate_mutexes: np.ndarray | list[float] | None = None,
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
        limited_candidates = candidate_paths[: self.num_candidates]
        precomputed_mutexes = None
        if candidate_mutexes is not None:
            precomputed_mutexes = np.asarray(candidate_mutexes, dtype=np.float64)
        fallback_mutexes: list[float | None] = [None] * len(limited_candidates)
        if precomputed_mutexes is None and mutex_enabled and limited_candidates and hasattr(
            future_mutex_detector, "compute_candidate_mutexes"
        ):
            batch_results = future_mutex_detector.compute_candidate_mutexes(
                all_paths, flow_id, limited_candidates, k
            )
            fallback_mutexes = [float(value) for value, _info in batch_results]
        for index, path in enumerate(limited_candidates, start=1):
            obs[index] = self.path_features(graph, current_path, path)
            mask[index] = obs[index, 5]
            if mutex_enabled and mask[index] > 0:
                if precomputed_mutexes is not None and index - 1 < len(precomputed_mutexes):
                    cached_mutex = float(precomputed_mutexes[index - 1])
                else:
                    cached_mutex = fallback_mutexes[index - 1]
                if cached_mutex is None:
                    cached_mutex, _ = future_mutex_detector.compute_candidate_mutex(
                        all_paths, flow_id, path, k
                    )
                candidate_mutex = float(cached_mutex)
                obs[index, 6] = candidate_mutex
                obs[index, 7] = keep_mutex - candidate_mutex
        return obs, mask
