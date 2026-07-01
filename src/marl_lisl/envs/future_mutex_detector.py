"""在短未来窗口内检测共享中继卫星的路径对互斥。"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from marl_lisl.utils.graph import encode_path_edges


class FutureMutexDetector:
    """计算有效路径两两共享中继节点产生的折扣冲突量。

    当前定义不涉及节点或链路容量：同一时隙内，任意两条完整有效路径只要
    共享至少一个被统计的卫星节点，就记 1 次路径互斥。同一对路径即使共享
    多个节点也仍只计 1 次，但这些共享节点都会记录到诊断信息中。
    """

    def __init__(
        self,
        graph_store,
        num_sats: int,
        future_window: int,
        future_discount: float = 0.95,
        include_source_dest_nodes: bool = False,
        path_cache_size: int = 200_000,
    ):
        self.graph_store = graph_store
        self.num_sats = int(num_sats)
        if self.num_sats <= 0:
            raise ValueError("num_sats must be positive")
        self.future_window = max(0, int(future_window))
        self.future_discount = float(future_discount)
        if not 0.0 <= self.future_discount <= 1.0:
            raise ValueError("future_discount must be in [0, 1]")
        self.include_source_dest_nodes = bool(include_source_dest_nodes)

        # 图边键只保留当前未来窗口，避免一次加载全部时隙；路径缓存则跨时隙
        # 保存边编码和节点集合，让大量候选动作共享同一份不可变路径画像。
        self._edge_key_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._edge_cache_size = self.future_window + 1
        self._path_cache_size = max(0, int(path_cache_size))
        self._path_cache: OrderedDict[
            tuple[int, ...] | None,
            tuple[np.ndarray, tuple[int, ...], frozenset[int]],
        ] = OrderedDict()

    def _edge_keys(self, k: int) -> np.ndarray:
        """读取一个图快照并返回可二分查找的无向边整数编码。"""
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
        while len(self._edge_key_cache) > self._edge_cache_size:
            self._edge_key_cache.popitem(last=False)
        return keys

    @staticmethod
    def _path_key(
        path: list[int] | tuple[int, ...] | None,
    ) -> tuple[int, ...] | None:
        """把路径转换成稳定缓存键；不足一条边的对象统一视为无效路径。"""
        if path is None or len(path) < 2:
            return None
        return tuple(int(node) for node in path)

    def _path_info(
        self, path: list[int] | tuple[int, ...] | None
    ) -> tuple[np.ndarray, tuple[int, ...], frozenset[int]]:
        """返回路径边编码、中继节点序列和用于求交集的节点集合。"""
        key = self._path_key(path)
        if key in self._path_cache:
            self._path_cache.move_to_end(key)
            return self._path_cache[key]
        if key is None:
            info = (np.empty(0, dtype=np.int64), (), frozenset())
        else:
            encoded = encode_path_edges(key, self.num_sats)
            nodes = key if self.include_source_dest_nodes else key[1:-1]
            occupied = tuple(
                int(node) for node in nodes if 0 <= int(node) < self.num_sats
            )
            info = (encoded, occupied, frozenset(occupied))
        if self._path_cache_size:
            self._path_cache[key] = info
            self._path_cache.move_to_end(key)
            while len(self._path_cache) > self._path_cache_size:
                self._path_cache.popitem(last=False)
        return info

    def _encoded_path_available(
        self, encoded: np.ndarray, edge_keys: np.ndarray
    ) -> bool:
        """判断路径的全部边是否同时存在于指定时隙。"""
        if not len(encoded):
            return False
        indices = np.searchsorted(edge_keys, encoded)
        return bool(
            np.all(indices < len(edge_keys))
            and np.all(edge_keys[indices] == encoded)
        )

    def _path_available(self, path: list[int] | None, edge_keys: np.ndarray) -> bool:
        """兼容单路径诊断调用，内部复用已经缓存的边编码。"""
        return self._encoded_path_available(self._path_info(path)[0], edge_keys)

    @staticmethod
    def _pair_conflicts(
        node_sets: list[frozenset[int]],
    ) -> tuple[int, set[int]]:
        """统计路径集合中的冲突路径对，并返回所有导致冲突的共享节点。

        这里按路径对计数而不是按共享节点数计数。例如两条路径同时共享节点
        10 和 11，结果仍为 1 次路径互斥，诊断节点集合则包含 ``{10, 11}``。
        """
        conflict_count = 0
        conflict_nodes: set[int] = set()
        for left in range(len(node_sets)):
            for right in range(left + 1, len(node_sets)):
                shared = node_sets[left].intersection(node_sets[right])
                if shared:
                    conflict_count += 1
                    conflict_nodes.update(shared)
        return conflict_count, conflict_nodes

    def compute_future_mutex(self, paths: list, k: int) -> tuple[float, dict]:
        """计算一组路径在未来窗口中的折扣路径对互斥。"""
        mutex_count = 0.0
        raw_conflict_count = 0
        invalid_future_path_count = 0
        first_conflict_slot: int | None = None
        first_conflict_nodes: list[int] = []
        evaluated_slots = 0
        path_infos = [self._path_info(path) for path in paths]

        for delta in range(self.future_window + 1):
            slot = int(k) + delta
            try:
                edge_keys = self._edge_keys(slot)
            except FileNotFoundError:
                if delta == 0:
                    raise
                break
            evaluated_slots += 1
            valid_node_sets: list[frozenset[int]] = []
            for encoded, _nodes, node_set in path_infos:
                if self._encoded_path_available(encoded, edge_keys):
                    valid_node_sets.append(node_set)
                else:
                    invalid_future_path_count += 1

            slot_conflicts, conflict_nodes = self._pair_conflicts(valid_node_sets)
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
        """一次扫描未来窗口，计算保持动作和所有单 flow 替换动作。

        对每个候选只重新计算“候选路径与其他有效路径”的交集；其他 flow
        之间的路径对冲突直接复用，因此不会为每个动作复制完整路径集合。
        """
        flow_count = len(paths)
        max_candidates = max(
            (len(candidates) for candidates in candidate_paths_by_flow), default=0
        )
        candidate_mutexes = np.zeros((flow_count, max_candidates), dtype=np.float64)
        keep_mutex = 0.0
        current_infos = [self._path_info(path) for path in paths]
        candidate_infos = [
            [self._path_info(path) for path in candidates]
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
            current_sets: list[frozenset[int] | None] = []
            for encoded, _nodes, node_set in current_infos:
                current_sets.append(
                    node_set
                    if self._encoded_path_available(encoded, edge_keys)
                    else None
                )
            keep_conflicts, _ = self._pair_conflicts(
                [node_set for node_set in current_sets if node_set is not None]
            )
            keep_mutex += discount * keep_conflicts

            for flow_id, candidates in enumerate(candidate_infos):
                other_sets = [
                    node_set
                    for index, node_set in enumerate(current_sets)
                    if index != flow_id and node_set is not None
                ]
                other_conflicts, _ = self._pair_conflicts(other_sets)
                for candidate_id, (encoded, _nodes, candidate_set) in enumerate(candidates):
                    slot_conflicts = other_conflicts
                    if self._encoded_path_available(encoded, edge_keys):
                        slot_conflicts += sum(
                            bool(candidate_set.intersection(other_set))
                            for other_set in other_sets
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
        """计算只替换一个 flow 后的完整路径互斥信息。"""
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
        """共享基础路径工作量，批量计算一个 flow 的全部候选互斥信息。"""
        flow_id = int(flow_id)
        candidate_count = len(candidate_paths)
        mutex_counts = np.zeros(candidate_count, dtype=np.float64)
        raw_conflict_counts = np.zeros(candidate_count, dtype=np.int64)
        invalid_counts = np.zeros(candidate_count, dtype=np.int64)
        first_slots: list[int | None] = [None] * candidate_count
        first_nodes: list[list[int]] = [[] for _ in range(candidate_count)]
        evaluated_slots = 0
        base_infos = [
            self._path_info(path)
            for index, path in enumerate(paths)
            if index != flow_id
        ]
        candidate_infos = [self._path_info(path) for path in candidate_paths]

        for delta in range(self.future_window + 1):
            slot = int(k) + delta
            try:
                edge_keys = self._edge_keys(slot)
            except FileNotFoundError:
                if delta == 0:
                    raise
                break
            evaluated_slots += 1
            base_sets: list[frozenset[int]] = []
            base_invalid = 0
            for encoded, _nodes, node_set in base_infos:
                if self._encoded_path_available(encoded, edge_keys):
                    base_sets.append(node_set)
                else:
                    base_invalid += 1
            base_conflicts, base_conflict_nodes = self._pair_conflicts(base_sets)

            for candidate_id, (encoded, _nodes, candidate_set) in enumerate(candidate_infos):
                candidate_valid = self._encoded_path_available(encoded, edge_keys)
                invalid_counts[candidate_id] += base_invalid + int(not candidate_valid)
                slot_conflicts = base_conflicts
                conflict_nodes = set(base_conflict_nodes)
                if candidate_valid:
                    for other_set in base_sets:
                        shared = candidate_set.intersection(other_set)
                        if shared:
                            slot_conflicts += 1
                            conflict_nodes.update(shared)
                if slot_conflicts:
                    raw_conflict_counts[candidate_id] += slot_conflicts
                    mutex_counts[candidate_id] += (
                        self.future_discount ** delta
                    ) * slot_conflicts
                    if first_slots[candidate_id] is None:
                        first_slots[candidate_id] = slot
                        first_nodes[candidate_id] = sorted(conflict_nodes)

        results: list[tuple[float, dict]] = []
        for candidate_id in range(candidate_count):
            value = float(mutex_counts[candidate_id])
            results.append(
                (
                    value,
                    {
                        "future_mutex": value,
                        "raw_conflict_count": int(raw_conflict_counts[candidate_id]),
                        "invalid_future_path_count": int(invalid_counts[candidate_id]),
                        "first_conflict_slot": first_slots[candidate_id],
                        "first_conflict_nodes": first_nodes[candidate_id],
                        "evaluated_slots": evaluated_slots,
                    },
                )
            )
        return results
