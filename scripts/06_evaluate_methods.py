#!/usr/bin/env python3
"""功能：统一评估 baseline、互斥诊断策略和可选的 MAPPO checkpoint。"""

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marl_lisl.algos.mappo import MAPPOTrainer
from marl_lisl.baselines import (
    MaintainUntilConflictPolicy,
    ProactiveRuleBaseline,
    ShortestDelayPolicy,
)
from marl_lisl.envs import LISLMultiFlowEnv
from marl_lisl.evaluation import Evaluator
from marl_lisl.evaluation.result_writer import (
    print_diagnostics,
    print_results_table,
    write_results_csv,
)
from marl_lisl.evaluation.script_utils import baseline_policies
from marl_lisl.utils.runtime_config import (
    load_runtime_env_config,
    load_runtime_mappo_config,
    resolve_project_path,
)


class MAPPODeterministicAdapter:
    """把 MAPPO 的确定性动作接口适配为通用 Evaluator 所需的 ``act``。"""

    def __init__(self, trainer: MAPPOTrainer):
        self.trainer = trainer

    def act(self, obs, state, action_mask):
        """使用 masked logits 的 argmax 返回每个 agent 的确定性动作。"""
        actions, *_ = self.trainer.policy.act(
            obs, state, action_mask, deterministic=True
        )
        return actions


def _policies(mode: str, config: dict) -> list[tuple[str, object]]:
    """按评估模式构造完整 baseline 或轻量 future-mutex 诊断策略集合。"""
    if mode == "diagnose":
        return [
            ("MaintainUntilConflict", MaintainUntilConflictPolicy()),
            ("ShortestDelay", ShortestDelayPolicy()),
            ("ProactiveRule", ProactiveRuleBaseline.from_config(config)),
        ]
    if mode in ("baselines", "all"):
        return baseline_policies(config)
    return []


def main() -> None:
    """运行所选方法，打印同口径诊断表并把结果保存为 CSV。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", type=Path, default=ROOT / "configs/env.yaml")
    parser.add_argument("--mappo-config", type=Path, default=ROOT / "configs/mappo.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--traffic", type=Path, required=True)
    parser.add_argument(
        "--methods",
        choices=("baselines", "diagnose", "mappo", "all"),
        default="all",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/tables/method_compare.csv"
    )
    parser.add_argument(
        "--preload-graphs", action="store_true", help="评估前预加载全部图，速度更快但占内存"
    )
    args = parser.parse_args()

    env_config = load_runtime_env_config(
        args.env_config,
        ROOT,
        traffic_path=args.traffic,
        preload_graphs=args.preload_graphs,
        train_random_start=False,
    )
    results: list[dict] = []
    for name, policy in _policies(args.methods, env_config):
        env = LISLMultiFlowEnv(env_config)
        results.append(
            {"method": name, **Evaluator(env, policy, args.max_steps).run_episode()}
        )

    if args.methods in ("mappo", "all"):
        if args.checkpoint is None:
            if args.methods == "mappo":
                parser.error("--methods mappo 必须提供 --checkpoint")
            warnings.warn("未提供 checkpoint，all 模式只评估 baseline。", stacklevel=2)
        else:
            checkpoint = resolve_project_path(ROOT, args.checkpoint)
            if not checkpoint.is_file():
                if args.methods == "mappo":
                    raise FileNotFoundError(f"MAPPO checkpoint 不存在: {checkpoint}")
                warnings.warn(f"checkpoint 不存在，跳过 MAPPO: {checkpoint}", stacklevel=2)
            else:
                mappo_config = load_runtime_mappo_config(args.mappo_config, ROOT)
                # 通用 Evaluator 只运行一个同步环境；强制 num_envs=1 防止评估时
                # 额外创建训练用子进程向量环境。
                mappo_config["num_envs"] = 1
                env = LISLMultiFlowEnv(env_config)
                trainer = MAPPOTrainer(env, mappo_config, env_config)
                try:
                    trainer.load_checkpoint(checkpoint)
                    policy = MAPPODeterministicAdapter(trainer)
                    results.append(
                        {
                            "method": "MAPPO",
                            **Evaluator(env, policy, args.max_steps).run_episode(),
                        }
                    )
                finally:
                    trainer.close()

    if not results:
        raise RuntimeError("没有可评估的方法，请检查 --methods 和 --checkpoint。")
    print_results_table(results)
    print_diagnostics(results)
    write_results_csv(results, args.output)


if __name__ == "__main__":
    main()
