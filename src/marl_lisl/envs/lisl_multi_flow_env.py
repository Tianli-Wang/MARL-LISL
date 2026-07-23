"""Minimal multi-agent, multi-flow routing environment over dynamic LISL graphs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from marl_lisl.envs.conflict_detector import ConflictDetector
from marl_lisl.envs.future_mutex_detector import FutureMutexDetector
from marl_lisl.envs.observation_builder import ObservationBuilder
from marl_lisl.envs.path_generator import PathGenerator
from marl_lisl.envs.reward_calculator import RewardCalculator
from marl_lisl.store.candidate_store import CandidateStore
from marl_lisl.store.graph_store import GraphStore
from marl_lisl.store.traffic_store import TrafficStore
from marl_lisl.utils.graph import edge_ids_for_node_path, edge_path_from_node_path


class LISLMultiFlowEnv:
    obs_dim = 8
    # state 的前 7 维必须永久保持原顺序，以便 state_dim=7 的历史 Critic
    # 仍可在新环境中截取这个前缀完成评估。
    legacy_state_dim = 7
    # 每条流不直接摊平全部候选，而是提取当前路径与两个候选 Pareto 端点的
    # 10 个摘要特征；这样随候选数增长时 Critic 输入维度仍保持可控。
    per_flow_state_dim = 10
    # 类属性仅作为旧代码的保守默认值；每个环境实例会在读取 num_flows 后
    # 覆盖为 ``7 + 10 * num_flows``。
    state_dim = legacy_state_dim

    @classmethod
    def state_size(cls, num_flows: int) -> int:
        """返回指定业务流数量对应的集中式 Critic state 维度。"""

        num_flows = int(num_flows)
        if num_flows <= 0:
            raise ValueError("num_flows 必须大于 0")
        return cls.legacy_state_dim + cls.per_flow_state_dim * num_flows

    def __init__(self, config: dict):
        self.config = config
        self.num_steps = int(config["num_steps"])
        self.num_sats = int(config["num_sats"])
        self.num_flows = int(config["num_flows"])
        self.state_dim = self.state_size(self.num_flows)
        self.num_candidates = int(config["num_candidates"])
        self.parallel_workers = max(1, int(config.get("parallel_workers", 1)))
        env_cfg = config["env"]
        self.episode_start = int(env_cfg["episode_start"])
        self.episode_length = int(env_cfg["episode_length"])
        self.invalid_action_penalty = float(env_cfg["action_invalid_penalty"])
        if not 0 <= self.episode_start < self.num_steps:
            raise ValueError("episode_start must be inside [0, num_steps)")
        self.episode_end = min(self.num_steps, self.episode_start + self.episode_length)

        mutex_cfg = config.get("future_mutex", {})
        graph_cache_size = int(
            config.get(
                "graph_cache_size",
                max(3, int(mutex_cfg.get("future_window", 0)) + 4),
            )
        )
        graph_backend = str(config.get("graph_backend", "lazy")).lower()
        if graph_backend == "packed":
            from marl_lisl.store.packed_graph_store import PackedGraphStore

            pack_dir = config.get("graph_pack_dir")
            self.graph_store: Any = PackedGraphStore(
                Path(config["graph_dir"]),
                pack_dir=Path(pack_dir) if pack_dir else None,
                cache_size=max(graph_cache_size, 32),
                build_if_missing=bool(config.get("graph_pack_build_if_missing", True)),
            )
        else:
            self.graph_store = GraphStore(
                Path(config["graph_dir"]),
                cache_size=graph_cache_size,
                preload=bool(config.get("graph_preload", False)),
            )
        traffic_path = config.get("traffic_path", config["traffic_train_path"])
        self.traffic_store = TrafficStore(Path(traffic_path))
        self.traffic_pairs = self.traffic_store.get_pairs()
        if len(self.traffic_pairs) != self.num_flows:
            raise ValueError(
                f"Configured num_flows={self.num_flows}, traffic file has {len(self.traffic_pairs)}"
            )
        self.path_generator = PathGenerator(self.num_candidates, config["path_weight"])
        candidates_cfg = dict(config.get("candidates", {}))
        self.use_precomputed_candidates = bool(candidates_cfg.get("enabled", False))
        # NPZ 与 packed memmap 后端暴露相同接口但没有共同基类，因此这里明确
        # 使用鸭子类型，避免类型检查器把变量错误限制为 CandidateStore。
        self.candidate_store: Any = None
        if self.use_precomputed_candidates:
            candidate_dir = config.get("candidate_dir")
            if candidate_dir is None:
                traffic_path_resolved = Path(traffic_path)
                eval_path = Path(config["traffic_eval_path"])
                train_path = Path(config["traffic_train_path"])
                stress_path = Path(config["traffic_stress_path"]) if "traffic_stress_path" in config else None
                if traffic_path_resolved == eval_path:
                    candidate_dir = candidates_cfg["eval_dir"]
                elif traffic_path_resolved == train_path:
                    candidate_dir = candidates_cfg["train_dir"]
                elif stress_path is not None and traffic_path_resolved == stress_path:
                    candidate_dir = candidates_cfg.get("stress_dir")
            if candidate_dir is not None:
                candidate_backend = str(candidates_cfg.get("backend", "npz")).lower()
                if candidate_backend == "packed":
                    from marl_lisl.store.packed_candidate_store import PackedCandidateStore

                    pack_dir = candidates_cfg.get("pack_dir")
                    self.candidate_store = PackedCandidateStore(
                        Path(candidate_dir),
                        self.num_flows,
                        self.num_candidates,
                        pack_dir=Path(pack_dir) if pack_dir else None,
                        cache_size=int(candidates_cfg.get("cache_size", 64)),
                        build_if_missing=bool(
                            candidates_cfg.get("pack_build_if_missing", True)
                        ),
                        expected_num_steps=self.num_steps,
                    )
                else:
                    self.candidate_store = CandidateStore(
                        Path(candidate_dir),
                        self.num_flows,
                        self.num_candidates,
                        cache_size=int(candidates_cfg.get("cache_size", 3)),
                    )
            else:
                self.use_precomputed_candidates = False
        self.observation_builder = ObservationBuilder(self.num_candidates, self.num_sats)
        self.reward_calculator = RewardCalculator(config["reward_weights"])
        self.conflict_detector = ConflictDetector(self.num_sats)
        self.future_mutex_enabled = bool(mutex_cfg.get("enabled", False))
        self.future_mutex_detector: FutureMutexDetector | None = None
        self.future_mutex_observation_detector: FutureMutexDetector | None = None
        if self.future_mutex_enabled:
            future_window = int(mutex_cfg["future_window"])
            observation_window = int(mutex_cfg.get("observation_window", future_window))
            path_cache_size = int(mutex_cfg.get("path_cache_size", 200_000))
            self.future_mutex_detector = FutureMutexDetector(
                self.graph_store,
                self.num_sats,
                future_window,
                float(mutex_cfg.get("future_discount", 0.95)),
                bool(mutex_cfg.get("include_source_dest_nodes", False)),
                path_cache_size=path_cache_size,
            )
            if observation_window == future_window:
                self.future_mutex_observation_detector = self.future_mutex_detector
            else:
                self.future_mutex_observation_detector = FutureMutexDetector(
                    self.graph_store,
                    self.num_sats,
                    observation_window,
                    float(mutex_cfg.get("future_discount", 0.95)),
                    bool(mutex_cfg.get("include_source_dest_nodes", False)),
                    path_cache_size=path_cache_size,
                )
        # Per-env RNG (seeded distinctly per worker) for scenario diversity.
        self._rng = np.random.default_rng(int(config.get("seed", 0)))
        # Random-start training: each reset begins at a different timeslot so the
        # policy sees the whole orbital trajectory rather than overfitting k=0.
        # Reuses the existing per-timeslot candidate files — no extra precompute.
        self.train_random_start = bool(env_cfg.get("train_random_start", False))
        self.train_min_steps = int(env_cfg.get("train_min_steps", self.episode_length))
        self._max_start = max(
            self.episode_start, self.num_steps - max(1, self.train_min_steps)
        )

        self.k = self.episode_start
        self.current_paths: list[list[int] | None] = [None] * self.num_flows
        self._candidate_paths: list[list[list[int]]] = [[] for _ in range(self.num_flows)]
        self._prepared_k: int | None = None
        self._prepared_cache: tuple | None = None
        self._done = False

    def _generate_candidates(self, graph: dict) -> list[list[list[int]]]:
        if self.candidate_store is not None:
            return self.candidate_store.get_candidates(self.k)
        pairs = [(int(source), int(dest)) for source, dest, _demand in self.traffic_pairs]
        self.path_generator.prepare_graph(graph)
        if self.parallel_workers <= 1 or len(pairs) <= 1:
            return [
                self.path_generator.generate(graph, source, dest)
                for source, dest in pairs
            ]
        worker_count = min(self.parallel_workers, len(pairs))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            return list(
                executor.map(
                    lambda item: self.path_generator.generate(graph, item[0], item[1]),
                    pairs,
                )
            )

    def _build_observations(
        self, graph: dict, candidates: list[list[list[int]]]
    ) -> tuple[np.ndarray, np.ndarray, float]:
        obs = np.zeros(
            (self.num_flows, self.num_candidates + 1, self.obs_dim), dtype=np.float32
        )
        masks = np.zeros((self.num_flows, self.num_candidates + 1), dtype=np.float32)
        candidate_mutexes = None
        observation_detector = self.future_mutex_observation_detector
        if observation_detector is not None and hasattr(
            observation_detector, "compute_flow_candidate_mutexes"
        ):
            future_mutex_keep, candidate_mutexes = (
                observation_detector.compute_flow_candidate_mutexes(
                    self.current_paths, candidates, self.k
                )
            )
        else:
            future_mutex_keep, _ = self._compute_future_mutex(self.current_paths, self.k)
        for flow_id in range(self.num_flows):
            obs[flow_id], masks[flow_id] = self.observation_builder.build_flow_obs(
                graph,
                self.current_paths[flow_id],
                candidates[flow_id],
                future_mutex_detector=observation_detector,
                all_paths=self.current_paths,
                flow_id=flow_id,
                k=self.k,
                future_mutex_keep=future_mutex_keep,
                candidate_mutexes=(
                    None if candidate_mutexes is None else candidate_mutexes[flow_id]
                ),
            )
        return obs, masks, future_mutex_keep

    def _compute_future_mutex(self, paths: list, k: int) -> tuple[float, dict]:
        if self.future_mutex_detector is None:
            return 0.0, {
                "future_mutex": 0.0,
                "raw_conflict_count": 0,
                "invalid_future_path_count": 0,
                "first_conflict_slot": None,
                "first_conflict_nodes": [],
                "evaluated_slots": 0,
            }
        return self.future_mutex_detector.compute_future_mutex(paths, k)

    def compute_current_future_mutex(self) -> tuple[float, dict]:
        """按 reward 使用的完整未来窗口计算当前路径保持动作的互斥量。

        observation 可以配置更短的 ``observation_window`` 来降低训练开销，
        因而不能直接用 state 最后一维与 step 后的完整 ``future_mutex`` 比较。
        该公共接口固定使用 ``future_mutex_detector``，保证规则诊断脚本中的
        keep/after 两个值采用同一窗口和同一折扣定义。
        """
        return self._compute_future_mutex(self.current_paths, self.k)

    def _build_state(
        self,
        graph: dict,
        obs: np.ndarray,
        mask: np.ndarray,
        future_mutex: float = 0.0,
    ) -> np.ndarray:
        """构建“全局 7 维 + 每流 10 维摘要”的 centralized state。

        每流候选摘要只统计 action 1..K 中 ``mask > 0`` 的真实可行候选。
        不可行候选的 observation 是全零，如果直接参与最小值会被误判成零时延、
        零互斥的最优路径，因此这里必须先用 mask 过滤。
        """

        expected_obs_shape = (
            self.num_flows,
            self.num_candidates + 1,
            self.obs_dim,
        )
        expected_mask_shape = (self.num_flows, self.num_candidates + 1)
        if obs.shape != expected_obs_shape or mask.shape != expected_mask_shape:
            raise ValueError(
                "构建 Critic state 时 observation/mask 形状不匹配: "
                f"obs 期望 {expected_obs_shape}、实际 {obs.shape}; "
                f"mask 期望 {expected_mask_shape}、实际 {mask.shape}"
            )

        attrs = graph["edge_attr"]
        summary = graph.get("_state_summary")
        if summary is None:
            summary = (
                int(graph["edge_index"].shape[1]),
                float(attrs[:, 0].mean()) if len(attrs) else 0.0,
                float(attrs[:, 1].mean()) if len(attrs) else 0.0,
                float(attrs[:, 3].mean()) if len(attrs) else 0.0,
            )
            graph["_state_summary"] = summary
        edge_count, mean_attr_0, mean_attr_1, mean_attr_3 = summary
        active = self.num_flows - self.conflict_detector.count_outages(
            self.current_paths, graph
        )
        global_state = np.asarray(
            (
                self.k / max(self.num_steps - 1, 1),
                edge_count,
                mean_attr_0,
                mean_attr_1,
                mean_attr_3,
                active,
                float(future_mutex),
            ),
            dtype=np.float32,
        )
        flow_state = np.zeros(
            (self.num_flows, self.per_flow_state_dim), dtype=np.float32
        )
        for flow_id in range(self.num_flows):
            keep = obs[flow_id, 0]
            # 当前路径状态：是否仍可行、传播时延、最短剩余寿命和跳数。
            # keep 的 setup/new-link 固定为 0，无需重复加入 state。
            flow_state[flow_id, 0:4] = (
                mask[flow_id, 0],
                keep[0],
                keep[2],
                keep[4],
            )

            candidate_mask = mask[flow_id, 1:] > 0.0
            # 正常配置至少有一个候选；仍显式兼容 K=0，避免空数组 mean 产生
            # NaN 并污染整个 state。
            flow_state[flow_id, 4] = (
                float(candidate_mask.mean()) if candidate_mask.size else 0.0
            )
            valid_indices = np.flatnonzero(candidate_mask)
            if valid_indices.size == 0:
                # legal ratio 已明确表明“没有可行候选”，其余字段保持 0 是稳定且
                # 可归一化的哨兵，不再用无效 observation 的零值参与 argmin。
                continue

            candidates = obs[flow_id, 1:][valid_indices]
            candidate_total_delay = candidates[:, 0] + candidates[:, 1]

            # 时延端点：保留最小时延候选本身的新链路数量，避免把来自不同动作的
            # 独立最小值拼成一个现实中不存在的“虚假最优候选”。
            delay_best = int(np.argmin(candidate_total_delay))
            flow_state[flow_id, 5] = candidate_total_delay[delay_best]
            flow_state[flow_id, 6] = candidates[delay_best, 3]

            # 互斥端点：同时保留该候选的总时延和新链路数量，让 Critic 能判断
            # 降低未来冲突需要付出多少时延与切换代价。
            mutex_best = int(np.argmin(candidates[:, 6]))
            flow_state[flow_id, 7] = candidates[mutex_best, 6]
            flow_state[flow_id, 8] = candidate_total_delay[mutex_best]
            flow_state[flow_id, 9] = candidates[mutex_best, 3]

        state = np.concatenate((global_state, flow_state.reshape(-1))).astype(
            np.float32, copy=False
        )
        if state.shape != (self.state_dim,) or not np.all(np.isfinite(state)):
            raise RuntimeError(
                "集中式 Critic state 构建失败: "
                f"期望 ({self.state_dim},)，实际 {state.shape}，"
                f"全为有限数={bool(np.all(np.isfinite(state)))}"
            )
        return state

    def _prepare_current(self) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
        """Build (and memoize) the current timeslot's graph/obs/state/mask.

        The expensive future-mutex observation pass is built once per timeslot and
        cached; previously ``step`` rebuilt it twice (once only to read the mask).
        """
        if self._prepared_cache is not None and self._prepared_k == self.k:
            return self._prepared_cache
        graph = self.graph_store.get_graph(self.k)
        self._candidate_paths = self._generate_candidates(graph)
        obs, mask, future_mutex = self._build_observations(graph, self._candidate_paths)
        state = self._build_state(graph, obs, mask, future_mutex)
        self._prepared_cache = (graph, obs, state, mask)
        self._prepared_k = self.k
        return self._prepared_cache

    def get_routing_context(self) -> tuple[dict, list, list]:
        """向需要路径级信息的规则 baseline 暴露当前只读路由上下文。

        普通学习策略只消费 observation；RSMR 还必须判断不同业务流的中继节点
        是否相交，因此需要真实节点路径。这里返回当前图、当前路径和候选路径，
        调用方只允许读取，不应原地修改这些环境内部对象。
        """
        graph, _obs, _state, _mask = self._prepare_current()
        return graph, self.current_paths, self._candidate_paths

    def edge_ids_for_path(self, graph: dict, path: list[int]) -> np.ndarray | None:
        """把节点路径映射为图中的边编号，供规则 baseline 复用统一图编码。"""
        return edge_ids_for_node_path(graph, path, self.num_sats)

    def reset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.train_random_start and self._max_start > self.episode_start:
            self.k = int(self._rng.integers(self.episode_start, self._max_start + 1))
        else:
            self.k = self.episode_start
        self.episode_end = min(self.num_steps, self.k + self.episode_length)
        self._done = False
        self._prepared_k = None
        self._prepared_cache = None
        self.current_paths = [None] * self.num_flows
        graph = self.graph_store.get_graph(self.k)
        self._candidate_paths = self._generate_candidates(graph)
        self.current_paths = [paths[0] if paths else None for paths in self._candidate_paths]
        obs, mask, future_mutex = self._build_observations(graph, self._candidate_paths)
        state = self._build_state(graph, obs, mask, future_mutex)
        self._prepared_cache = (graph, obs, state, mask)
        self._prepared_k = self.k
        return obs, state, mask

    def step(
        self, actions: np.ndarray | list[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool, dict]:
        if self._done:
            raise RuntimeError("Episode is done; call reset() before step()")
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.num_flows,):
            raise ValueError(f"actions must have shape ({self.num_flows},), got {actions.shape}")

        step_k = self.k
        graph, _obs, _state, action_mask = self._prepare_current()
        old_paths = [None if path is None else list(path) for path in self.current_paths]
        new_paths: list[list[int] | None] = []
        invalid_action_count = 0
        # 逐流保留非法动作标记，供路径导出时生成 Setup_Failures；汇总指标仍继续
        # 使用 invalid_action_count，因而不会改变已有 reward 和评估口径。
        invalid_actions = np.zeros(self.num_flows, dtype=bool)
        for flow_id, action in enumerate(actions.tolist()):
            if action < 0 or action > self.num_candidates or action_mask[flow_id, action] == 0:
                invalid_action_count += 1
                invalid_actions[flow_id] = True
                new_paths.append(None)
            elif action == 0:
                new_paths.append(old_paths[flow_id])
            else:
                new_paths.append(list(self._candidate_paths[flow_id][action - 1]))

        delays = np.zeros(self.num_flows, dtype=np.float64)
        feasible = np.zeros(self.num_flows, dtype=bool)
        switch_count = 0
        new_link_count = 0
        route_details: list[dict] = []
        for flow_id, (old_path, new_path) in enumerate(zip(old_paths, new_paths)):
            if new_path != old_path:
                switch_count += 1
            new_edges = edge_path_from_node_path(new_path) - edge_path_from_node_path(old_path)
            new_link_count += len(new_edges)
            features = self.observation_builder.path_features(graph, old_path, new_path)
            if features[5] == 1:
                feasible[flow_id] = True
                delays[flow_id] = float(features[0] + features[1])

            # 导出字段全部基于环境实际落地后的路径，而不是策略请求的动作。距离取
            # 当前路径所有 ISL 之和；建链惩罚沿用 observation 中“新增链路最大
            # setup delay”的定义。路径为空或不可行时记一次 setup failure。
            path_edge_ids = edge_ids_for_node_path(graph, new_path, self.num_sats)
            total_distance_km = (
                0.0
                if path_edge_ids is None
                else float(graph["edge_attr"][path_edge_ids, 0].sum() / 1000.0)
            )
            source, dest, _demand = self.traffic_pairs[flow_id]
            route_details.append(
                {
                    "flow_id": int(flow_id),
                    "source": int(source),
                    "target": int(dest),
                    "path": None if new_path is None else list(new_path),
                    "hops": 0 if new_path is None else max(0, len(new_path) - 1),
                    "total_isl_distance_km": total_distance_km,
                    "setup_penalty_ms": float(features[1] * 1000.0),
                    "setup_failures": int(
                        invalid_actions[flow_id] or not feasible[flow_id]
                    ),
                    "link_maintained": bool(new_path == old_path and new_path is not None),
                }
            )

        self.current_paths = new_paths
        outage_count = self.conflict_detector.count_outages(self.current_paths, graph)
        # ObservationBuilder 与 ConflictDetector 必须对当前图上的路径
        # 可行性得出完全相同的结论。该检查专门防止路径特征
        # 缓存串用时隙：如果观测认为路径可行，但真实边集
        # 检查认为已断链，立即终止并报出定位所需的时隙和数量。
        feature_outage_count = self.num_flows - int(feasible.sum())
        if feature_outage_count != outage_count:
            raise RuntimeError(
                "路径可行性检查不一致: "
                f"k={step_k}, path_features_outage={feature_outage_count}, "
                f"graph_outage={outage_count}"
            )
        future_mutex, future_mutex_info = self._compute_future_mutex(
            self.current_paths, step_k
        )
        # Infeasible/dropped flows are excluded from the delay average (the outage
        # term penalises them) so that dropping a flow can never lower avg_delay.
        reward, info = self.reward_calculator.compute(
            delays, switch_count, new_link_count, outage_count, future_mutex,
            feasible_mask=feasible,
        )
        invalid_penalty = self.invalid_action_penalty * invalid_action_count
        reward -= invalid_penalty
        info.update({
            "reward": float(reward),
            "invalid_action_count": int(invalid_action_count),
            "invalid_action_penalty": float(invalid_penalty),
            "delays": delays.copy(),
            "feasible_count": int(feasible.sum()),
            "k": step_k,
            "future_mutex": float(future_mutex),
            "future_mutex_info": future_mutex_info,
            "route_details": route_details,
        })

        self.k += 1
        self._prepared_k = None
        self._prepared_cache = None
        self._done = self.k >= self.episode_end
        if self._done:
            next_obs = np.zeros(
                (self.num_flows, self.num_candidates + 1, self.obs_dim), dtype=np.float32
            )
            next_state = np.zeros(self.state_dim, dtype=np.float32)
            next_mask = np.zeros((self.num_flows, self.num_candidates + 1), dtype=np.float32)
        else:
            _next_graph, next_obs, next_state, next_mask = self._prepare_current()
        return next_obs, next_state, next_mask, float(reward), self._done, info
