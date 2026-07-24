"""单 episode 的汇总与逐流路由评价指标。"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np


@dataclass
class _PerFlowAccumulator:
    """保存单条业务流在整个评估 episode 中的累计量。

    使用显式数据类而不是混合类型字典，既能避免字段名拼写错误，也能让
    mypy 明确区分标量与时延样本列表，便于后续继续扩展逐流指标。
    """

    flow_id: int
    source: int
    target: int
    num_steps: int = 0
    delays: list[float] = field(default_factory=list)
    delay_sum: float = 0.0
    peak_delay: float = 0.0
    outage_count: int = 0
    switch_count: int = 0
    new_link_count: int = 0
    maintain_count: int = 0
    setup_failure_count: int = 0
    hops_sum: float = 0.0
    hops_count: int = 0
    distance_sum_km: float = 0.0
    distance_count: int = 0
    setup_penalty_sum_ms: float = 0.0


class MetricsAccumulator:
    """累加时隙指标，并生成可跨流数量/轨迹长度比较的归一化结果。"""

    def __init__(self) -> None:
        self.total_reward = 0.0
        self.avg_delay_sum = 0.0
        self.peak_delay = 0.0
        self.future_mutex = 0.0
        self.outage_count = 0.0
        self.switch_count = 0.0
        self.new_link_count = 0.0
        self.invalid_action_count = 0.0
        self.num_steps = 0
        self.num_flows = 0
        self.raw_conflict_count = 0.0
        self.future_mutex_positive_steps = 0
        self.invalid_future_path_count = 0.0
        self.first_conflict_slot: int | None = None
        self.maintained_flow_steps = 0
        self.setup_failure_count = 0
        self.hops_sum = 0.0
        self.hops_count = 0
        self.path_distance_sum_km = 0.0
        self.path_distance_count = 0
        self.delay_samples: list[float] = []
        self.decision_times_s: list[float] = []
        # flow_id 可能来自外部自定义 traffic，不假设一定按首次 detail 顺序出现；
        # 字典使逐流统计对 route_details 排序保持鲁棒。
        self._per_flow: dict[int, _PerFlowAccumulator] = {}
        self._warned_missing: set[str] = set()

    def _value(self, info: dict, key: str, default: float = 0.0) -> float:
        if key not in info and key not in self._warned_missing:
            warnings.warn(f"info missing '{key}', using {default}", stacklevel=2)
            self._warned_missing.add(key)
        return float(info.get(key, default))

    def _flow_stats(self, detail: dict) -> _PerFlowAccumulator:
        """取得或初始化某条流的累计器，并保存固定源宿标识。"""

        flow_id = int(detail["flow_id"])
        if flow_id not in self._per_flow:
            self._per_flow[flow_id] = _PerFlowAccumulator(
                flow_id=flow_id,
                source=int(detail.get("source", -1)),
                target=int(detail.get("target", -1)),
            )
        return self._per_flow[flow_id]

    def update(
        self,
        reward: float,
        info: dict,
        *,
        decision_time_s: float | None = None,
    ) -> None:
        """加入一个环境时隙，并从 route_details 累加逐流路径质量。"""

        self.total_reward += float(reward)
        self.avg_delay_sum += self._value(info, "avg_delay")
        self.peak_delay = max(self.peak_delay, self._value(info, "peak_delay"))
        self.future_mutex += self._value(info, "future_mutex")
        self.outage_count += self._value(info, "outage_count")
        self.switch_count += self._value(info, "switch_count")
        self.new_link_count += self._value(info, "new_link_count")
        self.invalid_action_count += self._value(info, "invalid_action_count")
        self.num_steps += 1
        if decision_time_s is not None:
            decision_time_s = float(decision_time_s)
            if np.isfinite(decision_time_s) and decision_time_s >= 0.0:
                self.decision_times_s.append(decision_time_s)

        future_info = info.get("future_mutex_info")
        if isinstance(future_info, dict):
            raw_conflicts = float(future_info.get("raw_conflict_count", 0.0))
            self.raw_conflict_count += raw_conflicts
            self.invalid_future_path_count += float(
                future_info.get("invalid_future_path_count", 0.0)
            )
            conflict_slot = future_info.get("first_conflict_slot")
            if conflict_slot is not None:
                conflict_slot = int(conflict_slot)
                if (
                    self.first_conflict_slot is None
                    or conflict_slot < self.first_conflict_slot
                ):
                    self.first_conflict_slot = conflict_slot
        if float(info.get("future_mutex", 0.0)) > 0.0:
            # 这里统计的是“当前决策时隙的未来窗口存在冲突压力”，而不是把
            # future window 内每个冲突时隙重复计数。
            self.future_mutex_positive_steps += 1

        delays = np.asarray(info.get("delays", ()), dtype=np.float64).reshape(-1)
        route_details = info.get("route_details")
        if not isinstance(route_details, list):
            return
        self.num_flows = max(self.num_flows, len(route_details), len(delays))
        for detail in route_details:
            flow_id = int(detail["flow_id"])
            stats = self._flow_stats(detail)
            stats.num_steps += 1
            setup_failure = int(detail.get("setup_failures", 0)) > 0
            if setup_failure:
                stats.outage_count += 1
                stats.setup_failure_count += 1
                self.setup_failure_count += 1
            else:
                delay = float(delays[flow_id]) if flow_id < len(delays) else 0.0
                if np.isfinite(delay):
                    stats.delays.append(delay)
                    stats.delay_sum += delay
                    stats.peak_delay = max(stats.peak_delay, delay)
                    self.delay_samples.append(delay)

                hops = float(detail.get("hops", 0.0))
                distance_km = float(detail.get("total_isl_distance_km", 0.0))
                stats.hops_sum += hops
                stats.hops_count += 1
                stats.distance_sum_km += distance_km
                stats.distance_count += 1
                self.hops_sum += hops
                self.hops_count += 1
                self.path_distance_sum_km += distance_km
                self.path_distance_count += 1

            if bool(detail.get("switched", False)):
                stats.switch_count += 1
            flow_new_links = int(detail.get("new_link_count", 0))
            stats.new_link_count += flow_new_links
            if bool(detail.get("link_maintained", False)):
                stats.maintain_count += 1
                self.maintained_flow_steps += 1
            stats.setup_penalty_sum_ms += float(
                detail.get("setup_penalty_ms", 0.0)
            )

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        """为空样本提供稳定的 0，非空时返回 NumPy 线性分位数。"""

        if not values:
            return 0.0
        return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))

    def per_flow_summary(self) -> list[dict]:
        """返回每条流的时延、可靠性、切换和路径质量指标。"""

        rows: list[dict] = []
        for flow_id in sorted(self._per_flow):
            stats = self._per_flow[flow_id]
            steps = max(stats.num_steps, 1)
            delays = stats.delays
            delay_count = len(delays)
            hops_count = max(stats.hops_count, 1)
            distance_count = max(stats.distance_count, 1)
            rows.append(
                {
                    "flow_id": flow_id,
                    "source": stats.source,
                    "target": stats.target,
                    "avg_delay": (
                        stats.delay_sum / delay_count
                        if delay_count
                        else 0.0
                    ),
                    "delay_p95": self._percentile(delays, 95.0),
                    "peak_delay": stats.peak_delay,
                    "outage_count": stats.outage_count,
                    "outage_rate": stats.outage_count / steps,
                    "switch_count": stats.switch_count,
                    "switch_rate": stats.switch_count / steps,
                    "new_link_count": stats.new_link_count,
                    "new_link_rate": stats.new_link_count / steps,
                    "maintain_ratio": stats.maintain_count / steps,
                    "setup_failure_count": stats.setup_failure_count,
                    "setup_failure_rate": stats.setup_failure_count / steps,
                    "avg_hops": stats.hops_sum / hops_count,
                    "avg_path_distance_km": (
                        stats.distance_sum_km / distance_count
                    ),
                    "avg_setup_penalty_ms": (
                        stats.setup_penalty_sum_ms / steps
                    ),
                    "num_steps": stats.num_steps,
                }
            )
        return rows

    def summary(self) -> dict:
        """返回核心总量、分布指标和按 flow-step 归一化后的评价结果。"""

        flow_steps = max(self.num_steps * self.num_flows, 1)
        flow_rows = self.per_flow_summary()
        flow_avg_delays = [
            float(row["avg_delay"])
            for row in flow_rows
            if int(row["outage_count"]) < int(row["num_steps"])
        ]
        return {
            "total_reward": float(self.total_reward),
            "reward_per_step": float(
                self.total_reward / max(self.num_steps, 1)
            ),
            "avg_delay": float(self.avg_delay_sum / max(self.num_steps, 1)),
            "delay_p95": self._percentile(self.delay_samples, 95.0),
            "delay_p99": self._percentile(self.delay_samples, 99.0),
            "peak_delay": float(self.peak_delay),
            "worst_flow_avg_delay": (
                max(flow_avg_delays) if flow_avg_delays else 0.0
            ),
            "future_mutex": float(self.future_mutex),
            "future_mutex_per_step": float(
                self.future_mutex / max(self.num_steps, 1)
            ),
            "raw_conflict_count": float(self.raw_conflict_count),
            "future_mutex_positive_steps": int(
                self.future_mutex_positive_steps
            ),
            "future_mutex_positive_rate": float(
                self.future_mutex_positive_steps / max(self.num_steps, 1)
            ),
            "first_conflict_slot": (
                "" if self.first_conflict_slot is None
                else int(self.first_conflict_slot)
            ),
            "invalid_future_path_count": float(
                self.invalid_future_path_count
            ),
            "outage_count": float(self.outage_count),
            "outage_rate": float(self.outage_count / flow_steps),
            "switch_count": float(self.switch_count),
            "switch_rate": float(self.switch_count / flow_steps),
            "new_link_count": float(self.new_link_count),
            "new_link_rate": float(self.new_link_count / flow_steps),
            "maintain_ratio": float(self.maintained_flow_steps / flow_steps),
            "setup_failure_count": int(self.setup_failure_count),
            "setup_failure_rate": float(
                self.setup_failure_count / flow_steps
            ),
            "avg_hops": float(self.hops_sum / max(self.hops_count, 1)),
            "avg_path_distance_km": float(
                self.path_distance_sum_km
                / max(self.path_distance_count, 1)
            ),
            "mean_decision_time_ms": float(
                np.mean(self.decision_times_s) * 1000.0
                if self.decision_times_s
                else 0.0
            ),
            "p95_decision_time_ms": float(
                self._percentile(self.decision_times_s, 95.0) * 1000.0
            ),
            "invalid_action_count": float(self.invalid_action_count),
            "num_flows": int(self.num_flows),
            "num_steps": int(self.num_steps),
        }
