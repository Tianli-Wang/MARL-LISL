"""Discounted future relay-node mutex detection over a short graph window."""

from __future__ import annotations

from collections import Counter, OrderedDict

import numpy as np

from marl_lisl.utils.graph import encode_path_edges


class FutureMutexDetector:
    def __init__(
        self,
        graph_store,
        node_capacity: np.ndarray,
        future_window: int,
        future_discount: float = 0.95,
        include_source_dest_nodes: bool = False,
        path_cache_size: int = 200_000,
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
        self._path_cache_size = max(0, int(path_cache_size))
        self._path_cache: OrderedDict[
            tuple[int, ...] | None,
            tuple[np.ndarray, tuple[int, ...], Counter[int]],
        ] = OrderedDict()
        self._empty_counter: Counter[int] = Counter()

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
        return self._encoded_path_available(self._encoded_path_edges(path), edge_keys)

    def _path_key(self, path: list[int] | tuple[int, ...] | None) -> tuple[int, ...] | None:
        if path is None or len(path) < 2:
            return None
        return tuple(int(node) for node in path)

    def _path_info(
        self, path: list[int] | tuple[int, ...] | None
    ) -> tuple[np.ndarray, tuple[int, ...], Counter[int]]:
        key = self._path_key(path)
        if key in self._path_cache:
            self._path_cache.move_to_end(key)
            return self._path_cache[key]
        if key is None:
            info = (np.empty(0, dtype=np.int64), (), self._empty_counter)
        else:
            encoded = encode_path_edges(key, self.num_sats)
            nodes = key if self.include_source_dest_nodes else key[1:-1]
            occupied = tuple(node for node in nodes if 0 <= int(node) < self.num_sats)
            info = (encoded, occupied, Counter(occupied))
        if self._path_cache_size:
            self._path_cache[key] = info
            self._path_cache.move_to_end(key)
            while len(self._path_cache) > self._path_cache_size:
                self._path_cache.popitem(last=False)
        return info

    def _encoded_path_edges(self, path: list[int] | None) -> np.ndarray:
        """Encode a node path into sparse integer edge keys once for reuse."""
        return self._path_info(path)[0]

    def _encoded_path_available(self, encoded: np.ndarray, edge_keys: np.ndarray) -> bool:
        """Check whether all encoded path edges exist in one graph snapshot."""
        if not len(encoded):
            return False
        indices = np.searchsorted(edge_keys, encoded)
        return bool(np.all(indices < len(edge_keys)) and np.all(edge_keys[indices] == encoded))

    def _occupied_nodes(self, path: list[int]) -> list[int]:
        return list(self._path_info(path)[1])

    def compute_future_mutex(self, paths: list, k: int) -> tuple[float, dict]:
        mutex_count = 0.0
        raw_conflict_count = 0
        invalid_future_path_count = 0
        first_conflict_slot: int | None = None
        first_conflict_nodes: list[int] = []
        evaluated_slots = 0
        encoded_paths = [self._encoded_path_edges(path) for path in paths]
        occupied_nodes = [self._path_info(path)[1] for path in paths]

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
            for encoded, nodes in zip(encoded_paths, occupied_nodes):
                if not self._encoded_path_available(encoded, edge_keys):
                    invalid_future_path_count += 1
                    continue
                occupancy.update(nodes)

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

    def compute_flow_candidate_mutexes(
        self,
        paths: list,
        candidate_paths_by_flow: list[list[list[int] | None]],
        k: int,
    ) -> tuple[float, np.ndarray]:
        """Compute keep mutex and all single-flow replacement mutex values.

        This is the hot path for observations. It scans each future slot once,
        builds the current path occupancy once, then updates only nodes touched
        by a candidate replacement instead of copying a Counter for every action.
        """
        flow_count = len(paths)
        max_candidates = max(
            (len(candidates) for candidates in candidate_paths_by_flow),
            default=0,
        )
        candidate_mutexes = np.zeros((flow_count, max_candidates), dtype=np.float64)
        keep_mutex = 0.0

        current_encoded = [self._encoded_path_edges(path) for path in paths]
        current_infos = [self._path_info(path) for path in paths]
        current_nodes = [info[1] for info in current_infos]
        current_node_counts = [info[2] for info in current_infos]
        candidate_encoded = [
            [self._encoded_path_edges(path) for path in candidates]
            for candidates in candidate_paths_by_flow
        ]
        candidate_node_counts = [
            [
                self._path_info(path)[2] for path in candidates
            ]
            for candidates in candidate_paths_by_flow
        ]

        for delta in range(self.future_window + 1):
            slot = int(k) + delta
            try:
                edge_keys = self._edge_keys(slot)
            except FileNotFoundError:
                if delta == 0:
                    raise
                break

            discount = self.future_discount ** delta
            base_occupancy: Counter[int] = Counter()
            current_valid = np.zeros(flow_count, dtype=bool)
            for flow_id, (encoded, nodes) in enumerate(
                zip(current_encoded, current_nodes)
            ):
                if self._encoded_path_available(encoded, edge_keys):
                    current_valid[flow_id] = True
                    base_occupancy.update(nodes)

            base_conflicts = 0
            for node, count in base_occupancy.items():
                base_conflicts += max(0, count - int(self.node_capacity[node]))
            keep_mutex += discount * base_conflicts

            for flow_id in range(flow_count):
                old_counts = (
                    current_node_counts[flow_id]
                    if current_valid[flow_id]
                    else Counter()
                )
                for candidate_id, encoded in enumerate(candidate_encoded[flow_id]):
                    candidate_valid = self._encoded_path_available(encoded, edge_keys)
                    new_counts = (
                        candidate_node_counts[flow_id][candidate_id]
                        if candidate_valid
                        else Counter()
                    )
                    touched_nodes = set(old_counts) | set(new_counts)
                    slot_conflicts = base_conflicts
                    for node in touched_nodes:
                        before = int(base_occupancy.get(node, 0))
                        after = (
                            before
                            - int(old_counts.get(node, 0))
                            + int(new_counts.get(node, 0))
                        )
                        capacity = int(self.node_capacity[node])
                        slot_conflicts += max(0, after - capacity) - max(
                            0, before - capacity
                        )
                    candidate_mutexes[flow_id, candidate_id] += (
                        discount * slot_conflicts
                    )

        return float(keep_mutex), candidate_mutexes

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

    def compute_candidate_mutexes(
        self,
        paths: list,
        flow_id: int,
        candidate_paths: list[list[int] | None],
        k: int,
    ) -> list[tuple[float, dict]]:
        """Compute future mutex for many replacement paths with shared slot work."""
        flow_id = int(flow_id)
        candidate_count = len(candidate_paths)
        mutex_counts = np.zeros(candidate_count, dtype=np.float64)
        raw_conflict_counts = np.zeros(candidate_count, dtype=np.int64)
        invalid_counts = np.zeros(candidate_count, dtype=np.int64)
        first_slots: list[int | None] = [None] * candidate_count
        first_nodes: list[list[int]] = [[] for _ in range(candidate_count)]
        evaluated_slots = 0

        base_encoded: list[np.ndarray] = []
        base_nodes: list[list[int]] = []
        for index, path in enumerate(paths):
            if index == flow_id:
                continue
            base_encoded.append(self._encoded_path_edges(path))
            base_nodes.append(self._occupied_nodes(path) if path is not None else [])
        candidate_encoded = [self._encoded_path_edges(path) for path in candidate_paths]
        candidate_nodes = [
            self._occupied_nodes(path) if path is not None else []
            for path in candidate_paths
        ]

        for delta in range(self.future_window + 1):
            slot = int(k) + delta
            try:
                edge_keys = self._edge_keys(slot)
            except FileNotFoundError:
                if delta == 0:
                    raise
                break
            evaluated_slots += 1

            base_occupancy: Counter[int] = Counter()
            base_invalid = 0
            for encoded, nodes in zip(base_encoded, base_nodes):
                if not self._encoded_path_available(encoded, edge_keys):
                    base_invalid += 1
                    continue
                base_occupancy.update(nodes)

            for candidate_id, (encoded, nodes) in enumerate(
                zip(candidate_encoded, candidate_nodes)
            ):
                occupancy = base_occupancy.copy()
                invalid = base_invalid
                if self._encoded_path_available(encoded, edge_keys):
                    occupancy.update(nodes)
                else:
                    invalid += 1
                invalid_counts[candidate_id] += invalid

                conflict_nodes: list[int] = []
                slot_conflicts = 0
                for node, count in occupancy.items():
                    excess = count - int(self.node_capacity[node])
                    if excess > 0:
                        slot_conflicts += excess
                        conflict_nodes.append(node)
                if slot_conflicts:
                    raw_conflict_counts[candidate_id] += slot_conflicts
                    mutex_counts[candidate_id] += (self.future_discount ** delta) * slot_conflicts
                    if first_slots[candidate_id] is None:
                        first_slots[candidate_id] = slot
                        first_nodes[candidate_id] = sorted(conflict_nodes)

        results: list[tuple[float, dict]] = []
        for candidate_id in range(candidate_count):
            value = float(mutex_counts[candidate_id])
            results.append((
                value,
                {
                    "future_mutex": value,
                    "raw_conflict_count": int(raw_conflict_counts[candidate_id]),
                    "invalid_future_path_count": int(invalid_counts[candidate_id]),
                    "first_conflict_slot": first_slots[candidate_id],
                    "first_conflict_nodes": first_nodes[candidate_id],
                    "evaluated_slots": evaluated_slots,
                },
            ))
        return results
