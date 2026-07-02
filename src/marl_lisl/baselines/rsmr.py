"""RSMR（反应式逐流节点遮罩路由）baseline。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from marl_lisl.envs import LISLMultiFlowEnv


class RSMRPolicy:
    """按固定流顺序执行“可保持则保持，否则重路由”的节点互斥策略。

    当前环境把每条流的动作空间表示为“保持动作 + K 条候选路径”，因此这里的
    节点遮罩最短路是在该时隙已经生成的 K 条候选路径中求解。这样既遵守环境的
    离散动作接口，也保证 baseline 与 MAPPO 使用完全相同的候选路径集合。
    """

    def __init__(self, path_weight: dict | None = None):
        # 路径搜索阶段使用传播、建链和寿命风险三项综合代价。策略选择候选时沿用
        # 同一组系数，避免仅按观测中的传播时延选择而偏离论文里的 c_e 定义。
        cfg = path_weight or {}
        self.propagation_weight = float(cfg.get("propagation", 1.0))
        self.setup_weight = float(cfg.get("setup", 1.0))
        self.lifetime_weight = float(cfg.get("lifetime", 0.1))
        self.lifetime_epsilon = float(cfg.get("lifetime_epsilon", 1.0))
        self._env: LISLMultiFlowEnv | None = None

    @classmethod
    def from_config(cls, config: dict) -> "RSMRPolicy":
        """从环境配置读取与候选路径生成器一致的边权参数。"""
        return cls(config.get("path_weight", {}))

    def bind_env(self, env: "LISLMultiFlowEnv") -> None:
        """绑定正在评估的环境，以读取候选路径及其真实中继节点集合。"""
        self._env = env

    @staticmethod
    def _relay_nodes(path: list[int] | None) -> set[int]:
        """返回路径中继节点；源、宿节点按文档定义不参与节点互斥。"""
        if path is None or len(path) <= 2:
            return set()
        return {int(node) for node in path[1:-1]}

    def _path_cost(self, graph: dict, path: list[int]) -> float:
        """计算文档定义的逐边综合代价之和；不存在的边返回无穷大。"""
        edge_ids = self._env.edge_ids_for_path(graph, path) if self._env else None
        if edge_ids is None:
            return float("inf")
        attrs = graph["edge_attr"][edge_ids]
        return float(
            self.propagation_weight * attrs[:, 1].sum()
            + self.setup_weight * attrs[:, 2].sum()
            + self.lifetime_weight
            * np.sum(1.0 / (attrs[:, 3] + self.lifetime_epsilon))
        )

    def act(self, obs, state, action_mask):  # noqa: D401 - 统一策略接口
        """按流编号顺序保留可行旧路，或选择不占用既有中继的最低代价候选。"""
        if self._env is None:
            raise RuntimeError("RSMRPolicy 必须由 Evaluator 绑定环境后才能执行")

        action_mask = np.asarray(action_mask)
        graph, current_paths, candidate_paths = self._env.get_routing_context()
        actions = np.zeros(len(current_paths), dtype=np.int64)
        occupied_relays: set[int] = set()

        for flow_id, current_path in enumerate(current_paths):
            # 动作 0 同时满足当前拓扑可行性和此前流的节点遮罩时，RSMR 必须保持，
            # 不能因为另有更短候选而主动切换。
            current_relays = self._relay_nodes(current_path)
            if (
                action_mask[flow_id, 0] > 0
                and current_relays.isdisjoint(occupied_relays)
            ):
                occupied_relays.update(current_relays)
                continue

            # 候选动作 i 对应 candidate_paths[i - 1]。先应用环境拓扑遮罩，再应用
            # RSMR 逐流累积的中继节点遮罩，最后按综合边权选择最优路径。
            feasible: list[tuple[float, int, set[int]]] = []
            for action in np.flatnonzero(action_mask[flow_id] > 0):
                if action == 0 or action - 1 >= len(candidate_paths[flow_id]):
                    continue
                path = candidate_paths[flow_id][action - 1]
                relays = self._relay_nodes(path)
                if relays.isdisjoint(occupied_relays):
                    feasible.append((self._path_cost(graph, path), int(action), relays))

            if feasible:
                # action 编号作为最终稳定 tie-break，保证相同输入下结果可复现。
                _cost, selected_action, selected_relays = min(
                    feasible, key=lambda item: (item[0], item[1])
                )
                actions[flow_id] = selected_action
                occupied_relays.update(selected_relays)
            else:
                # 没有满足节点遮罩的候选时输出 0。若保持动作本身非法，环境会把该
                # 流记为 outage；这与文档中的空路径语义一致，并保留非法动作诊断。
                actions[flow_id] = 0

        return actions
