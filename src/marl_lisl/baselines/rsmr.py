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

    def __init__(
        self,
        path_weight: dict | None = None,
        *,
        max_detour_ratio: float = 1.5,
        max_extra_hops: int = 2,
    ):
        # 路径搜索阶段使用传播、建链和寿命风险三项综合代价。策略选择候选时沿用
        # 同一组系数，避免仅按观测中的传播时延选择而偏离论文里的 c_e 定义。
        cfg = path_weight or {}
        self.propagation_weight = float(cfg.get("propagation", 1.0))
        self.setup_weight = float(cfg.get("setup", 1.0))
        self.lifetime_weight = float(cfg.get("lifetime", 0.1))
        self.lifetime_epsilon = float(cfg.get("lifetime_epsilon", 1.0))
        if max_detour_ratio < 1.0:
            raise ValueError("RSMR max_detour_ratio 不能小于 1.0")
        if max_extra_hops < 0:
            raise ValueError("RSMR max_extra_hops 不能小于 0")
        # 绕路上限同时约束物理距离和跳数。距离比例允许为了链路寿命、建链代价
        # 或节点互斥进行适当绕行，额外跳数则阻止大量短链路拼成超长蛇形路径。
        self.max_detour_ratio = float(max_detour_ratio)
        self.max_extra_hops = int(max_extra_hops)
        self._env: LISLMultiFlowEnv | None = None

    @classmethod
    def from_config(cls, config: dict) -> "RSMRPolicy":
        """从环境配置读取与候选路径生成器一致的边权参数。"""
        cfg = config.get("rsmr", {})
        return cls(
            config.get("path_weight", {}),
            max_detour_ratio=cfg.get("max_detour_ratio", 1.5),
            max_extra_hops=cfg.get("max_extra_hops", 2),
        )

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

    def _path_distance(self, graph: dict, path: list[int]) -> float:
        """返回路径物理距离（米）；边已失效时返回无穷大。"""
        edge_ids = self._env.edge_ids_for_path(graph, path) if self._env else None
        if edge_ids is None:
            return float("inf")
        return float(graph["edge_attr"][edge_ids, 0].sum())

    def _detour_allowed(
        self,
        distance: float,
        hops: int,
        shortest_distance: float,
        minimum_hops: int,
    ) -> bool:
        """判断路径是否同时满足距离比例和额外跳数两项绕路限制。"""
        return (
            distance <= shortest_distance * self.max_detour_ratio
            and hops <= minimum_hops + self.max_extra_hops
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
            # 绕路基准只使用当前拓扑合法候选，但暂不应用前序流节点遮罩。这样基准
            # 表示该源宿对本来可以达到的最短尺度，不会被互斥造成的坏候选抬高。
            route_metrics: dict[int, tuple[float, int]] = {}
            for action in np.flatnonzero(action_mask[flow_id] > 0):
                if action == 0 or action - 1 >= len(candidate_paths[flow_id]):
                    continue
                path = candidate_paths[flow_id][action - 1]
                route_metrics[int(action)] = (
                    self._path_distance(graph, path),
                    max(0, len(path) - 1),
                )
            reference_metrics = list(route_metrics.values())
            if current_path is not None and action_mask[flow_id, 0] > 0:
                reference_metrics.append(
                    (self._path_distance(graph, current_path), len(current_path) - 1)
                )
            shortest_distance = min(
                (item[0] for item in reference_metrics), default=float("inf")
            )
            minimum_hops = min((item[1] for item in reference_metrics), default=0)

            # 动作 0 同时满足当前拓扑可行性和此前流的节点遮罩时，RSMR 必须保持，
            # 且没有形成严重绕路时才保持。超过阈值视为路径质量失效并触发重路由。
            current_relays = self._relay_nodes(current_path)
            current_distance = (
                float("inf") if current_path is None
                else self._path_distance(graph, current_path)
            )
            if (
                action_mask[flow_id, 0] > 0
                and current_relays.isdisjoint(occupied_relays)
                and self._detour_allowed(
                    current_distance,
                    0 if current_path is None else len(current_path) - 1,
                    shortest_distance,
                    minimum_hops,
                )
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
                distance, hops = route_metrics[int(action)]
                if (
                    relays.isdisjoint(occupied_relays)
                    and self._detour_allowed(
                        distance, hops, shortest_distance, minimum_hops
                    )
                ):
                    feasible.append((self._path_cost(graph, path), int(action), relays))

            if feasible:
                # action 编号作为最终稳定 tie-break，保证相同输入下结果可复现。
                _cost, selected_action, selected_relays = min(
                    feasible, key=lambda item: (item[0], item[1])
                )
                actions[flow_id] = selected_action
                occupied_relays.update(selected_relays)
            else:
                # 没有同时满足节点遮罩与绕路上限的候选时显式输出非法动作 -1，
                # 环境会把路径置空并记为 outage。不能退回动作 0，否则在保持动作
                # 拓扑合法但节点冲突/严重绕路时，会悄悄违反 RSMR 的约束。
                actions[flow_id] = -1

        return actions
