"""Baseline 注册表：集中提供评估入口所需的方法发现与实例化能力。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .greedy_conflict_aware import GreedyConflictAwarePolicy
from .maintain_until_conflict import MaintainUntilConflictPolicy
from .proactive_rule_baseline import ProactiveRuleBaseline
from .rsmr import RSMRPolicy
from .shortest_delay import ShortestDelayPolicy


@dataclass(frozen=True)
class BaselineSpec:
    """描述一种 baseline，并保存它的名称、构造器及诊断分组信息。"""

    name: str
    factory: Callable[[dict], object]
    diagnostic: bool = False

    def create(self, config: dict) -> object:
        """根据本次评估的环境配置创建一个全新的策略实例。"""
        return self.factory(config)


# baseline 的唯一总清单：评估主脚本不再直接依赖任何具体策略类。
BASELINE_SPECS: tuple[BaselineSpec, ...] = (
    BaselineSpec("RSMR", RSMRPolicy.from_config),
    BaselineSpec("ShortestDelay", lambda _config: ShortestDelayPolicy(), True),
    BaselineSpec("MaintainUntilConflict", lambda _config: MaintainUntilConflictPolicy(), True),
    BaselineSpec("GreedyConflictAware", lambda _config: GreedyConflictAwarePolicy()),
    BaselineSpec("ProactiveRule", ProactiveRuleBaseline.from_config, True),
)


def build_baseline_policies(
    config: dict, *, diagnostic_only: bool = False
) -> list[tuple[str, object]]:
    """实例化 baseline；诊断模式仅返回标记为 diagnostic 的轻量方法。"""
    return [
        (spec.name, spec.create(config))
        for spec in BASELINE_SPECS
        if not diagnostic_only or spec.diagnostic
    ]
