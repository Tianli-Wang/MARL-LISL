"""Minimal multi-agent, multi-flow routing environment over dynamic LISL graphs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from marl_lisl.envs.conflict_detector import ConflictDetector
from marl_lisl.envs.future_mutex_detector import FutureMutexDetector
from marl_lisl.envs.observation_builder import ObservationBuilder
from marl_lisl.envs.path_generator import PathGenerator
from marl_lisl.envs.reward_calculator import RewardCalculator
from marl_lisl.store.candidate_store import CandidateStore
from marl_lisl.store.graph_store import GraphStore
from marl_lisl.store.mutex_store import MutexStore
from marl_lisl.store.traffic_store import TrafficStore
from marl_lisl.utils.graph import edge_path_from_node_path


class LISLMultiFlowEnv:
    obs_dim = 8
    state_dim = 7

    def __init__(self, config: dict):
        self.config = config
        self.num_steps = int(config["num_steps"])
        self.num_sats = int(config["num_sats"])
        self.num_flows = int(config["num_flows"])
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
            self.graph_store = PackedGraphStore(
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
        self.candidate_store: CandidateStore | None = None
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
        self.mutex_store: MutexStore | None = None
        self.future_mutex_detector: FutureMutexDetector | None = None
        self.future_mutex_observation_detector: FutureMutexDetector | None = None
        if self.future_mutex_enabled:
            self.mutex_store = MutexStore(Path(mutex_cfg["node_mutex_path"]))
            node_capacity = self.mutex_store.get_node_capacity()
            if len(node_capacity) != self.num_sats:
                raise ValueError(
                    f"node mutex length {len(node_capacity)} != num_sats {self.num_sats}"
                )
            future_window = int(mutex_cfg["future_window"])
            observation_window = int(mutex_cfg.get("observation_window", future_window))
            path_cache_size = int(mutex_cfg.get("path_cache_size", 200_000))
            self.future_mutex_detector = FutureMutexDetector(
                self.graph_store,
                node_capacity,
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
                    node_capacity,
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

    def _build_state(self, graph: dict, future_mutex: float = 0.0) -> np.ndarray:
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
        return np.asarray(
            [
                self.k / max(self.num_steps - 1, 1),
                edge_count,
                mean_attr_0,
                mean_attr_1,
                mean_attr_3,
                active,
                float(future_mutex),
            ],
            dtype=np.float32,
        )

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
        state = self._build_state(graph, future_mutex)
        self._prepared_cache = (graph, obs, state, mask)
        self._prepared_k = self.k
        return self._prepared_cache

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
        state = self._build_state(graph, future_mutex)
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
        for flow_id, action in enumerate(actions.tolist()):
            if action < 0 or action > self.num_candidates or action_mask[flow_id, action] == 0:
                invalid_action_count += 1
                new_paths.append(None)
            elif action == 0:
                new_paths.append(old_paths[flow_id])
            else:
                new_paths.append(list(self._candidate_paths[flow_id][action - 1]))

        delays = np.zeros(self.num_flows, dtype=np.float64)
        feasible = np.zeros(self.num_flows, dtype=bool)
        switch_count = 0
        new_link_count = 0
        for flow_id, (old_path, new_path) in enumerate(zip(old_paths, new_paths)):
            if new_path != old_path:
                switch_count += 1
            new_edges = edge_path_from_node_path(new_path) - edge_path_from_node_path(old_path)
            new_link_count += len(new_edges)
            features = self.observation_builder.path_features(graph, old_path, new_path)
            if features[5] == 1:
                feasible[flow_id] = True
                delays[flow_id] = float(features[0] + features[1])

        self.current_paths = new_paths
        outage_count = self.conflict_detector.count_outages(self.current_paths, graph)
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
