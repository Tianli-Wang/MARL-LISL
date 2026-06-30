"""Discounted future relay-node mutex detection over a short graph window."""

from __future__ import annotations

from collections import Counter, OrderedDict

import numpy as np

from marl_lisl.utils.graph import edge_path_from_node_path


class FutureMutexDetector:
    def __init__(
        self,
        graph_store,
        node_capacity: np.ndarray,
        future_window: int,
        future_discount: float = 0.95,
        include_source_dest_nodes: bool = False,
    ):
        self.graph_store = graph_store
        self.node_capacity = np.asarray(node_capacity, dtype=np.int32)
        if self.node_capacity.ndim != 1 or np.any(self.node_capacity < 1):
            raise ValueError("node_capacity must be a positive one-dimensional array")
        self.num_sats = len(self.node_capacity)
        self.future_window = max(0, int(future_window))
        self.future_discount = float(future_discount)
        if not 0.0 <= self.future_discount <= 1.0:
            raise ValueError("future_discount must be in [0, 1]")
        self.include_source_dest_nodes = bool(include_source_dest_nodes)
        # Only encoded sparse edge keys for the active future window are retained.
        self._edge_key_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_size = self.future_window + 1

    def _edge_keys(self, k: int) -> np.ndarray:
        if k in self._edge_key_cache:
            self._edge_key_cache.move_to_end(k)
            return self._edge_key_cache[k]
        graph = self.graph_store.get_graph(k)
        edge_index = graph["edge_index"]
        low = np.minimum(edge_index[0], edge_index[1]).astype(np.int64)
        high = np.maximum(edge_index[0], edge_index[1]).astype(np.int64)
        keys = low * self.num_sats + high
        if len(keys) > 1 and np.any(keys[1:] < keys[:-1]):
            keys = np.sort(keys)
        self._edge_key_cache[k] = keys
        while len(self._edge_key_cache) > self._cache_size:
            self._edge_key_cache.popitem(last=False)
        return keys

    def _path_available(self, path: list[int] | None, edge_keys: np.ndarray) -> bool:
        if path is None or len(path) < 2:
            return False
        encoded = np.fromiter(
            (u * self.num_sats + v for u, v in edge_path_from_node_path(path)),
            dtype=np.int64,
        )
        if not len(encoded):
            return False
        indices = np.searchsorted(edge_keys, encoded)
        return bool(np.all(indices < len(edge_keys)) and np.all(edge_keys[indices] == encoded))

    def _occupied_nodes(self, path: list[int]) -> list[int]:
        nodes = path if self.include_source_dest_nodes else path[1:-1]
        return [int(node) for node in nodes if 0 <= int(node) < self.num_sats]

    def compute_future_mutex(self, paths: list, k: int) -> tuple[float, dict]:
        mutex_count = 0.0
        raw_conflict_count = 0
        invalid_future_path_count = 0
        first_conflict_slot: int | None = None
        first_conflict_nodes: list[int] = []
        evaluated_slots = 0

        for delta in range(self.future_window + 1):
            slot = int(k) + delta
            try:
                edge_keys = self._edge_keys(slot)
            except FileNotFoundError:
                if delta == 0:
                    raise
                break
            evaluated_slots += 1
            occupancy: Counter[int] = Counter()
            for path in paths:
                if not self._path_available(path, edge_keys):
                    invalid_future_path_count += 1
                    continue
                occupancy.update(self._occupied_nodes(path))

            conflict_nodes: list[int] = []
            slot_conflicts = 0
            for node, count in occupancy.items():
                excess = count - int(self.node_capacity[node])
                if excess > 0:
                    slot_conflicts += excess
                    conflict_nodes.append(node)
            if slot_conflicts:
                raw_conflict_count += slot_conflicts
                mutex_count += (self.future_discount ** delta) * slot_conflicts
                if first_conflict_slot is None:
                    first_conflict_slot = slot
                    first_conflict_nodes = sorted(conflict_nodes)

        info = {
            "future_mutex": float(mutex_count),
            "raw_conflict_count": int(raw_conflict_count),
            "invalid_future_path_count": int(invalid_future_path_count),
            "first_conflict_slot": first_conflict_slot,
            "first_conflict_nodes": first_conflict_nodes,
            "evaluated_slots": evaluated_slots,
        }
        return float(mutex_count), info

    def compute_candidate_mutex(
        self,
        paths: list,
        flow_id: int,
        candidate_path: list[int] | None,
        k: int,
    ) -> tuple[float, dict]:
        new_paths = list(paths)
        new_paths[int(flow_id)] = candidate_path
        return self.compute_future_mutex(new_paths, k)
