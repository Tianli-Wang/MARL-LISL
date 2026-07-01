#!/usr/bin/env python3
"""功能：统一评估 baseline、互斥诊断策略和可选的 MAPPO checkpoint。"""

from __future__ import annotations

import argparse
import sys
import warnings
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# 默认数据与模型均选择当前仓库中已经生成并通过烟测的文件，使脚本无需参数
# 即可完成 baseline + MAPPO 同口径评估；命令行参数仍可覆盖任意一项。
DEFAULT_ENV_CONFIG = ROOT / "configs/env.yaml"
DEFAULT_MAPPO_CONFIG = ROOT / "configs/mappo.yaml"
DEFAULT_CHECKPOINT = (
    ROOT
    / "outputs/runs/20260701_175147_mappo_a100_16flow_vec/checkpoints/latest.pt"
)
DEFAULT_TRAFFIC = ROOT / "data/traffic/traffic_pairs_eval.npy"
DEFAULT_OUTPUT = ROOT / "outputs/tables/method_compare.csv"

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
    load_checkpoint_mappo_config,
    load_runtime_env_config,
    load_runtime_mappo_config,
    resolve_project_path,
)


class MAPPODeterministicAdapter:
    """把 MAPPO 的确定性动作接口适配为通用 Evaluator 所需的 ``act``。"""

    def __init__(self, trainer):
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
    parser.add_argument(
        "--env-config", type=Path, default=DEFAULT_ENV_CONFIG,
        help=f"环境配置，默认：{DEFAULT_ENV_CONFIG}",
    )
    parser.add_argument(
        "--mappo-config", type=Path, default=DEFAULT_MAPPO_CONFIG,
        help=f"MAPPO 配置，默认：{DEFAULT_MAPPO_CONFIG}",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help=f"MAPPO checkpoint，默认：{DEFAULT_CHECKPOINT}",
    )
    parser.add_argument(
        "--traffic", type=Path, default=DEFAULT_TRAFFIC,
        help=f"评估 traffic，默认：{DEFAULT_TRAFFIC}",
    )
    parser.add_argument(
        "--methods",
        choices=("baselines", "diagnose", "mappo", "all"),
        default="all",
        help="评估方法集合，默认：all",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="每种方法最多运行多少步，默认：完整 episode",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"结果 CSV，默认：{DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--preload-graphs", action="store_true", default=False,
        help="评估前预加载全部图，默认关闭；开启后速度更快但占内存",
    )
    args = parser.parse_args()

    env_config = load_runtime_env_config(
        args.env_config,
        ROOT,
        traffic_path=args.traffic,
        preload_graphs=args.preload_graphs,
        train_random_start=False,
    )
    output_path = resolve_project_path(ROOT, args.output)
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
                # baseline-only 模式不会导入 PyTorch；只有确实评估 MAPPO 时才
                # 加载训练器，减少启动时间并避免无关 CUDA/NumPy 环境错误。
                from marl_lisl.algos.mappo import MAPPOTrainer

                yaml_config = load_runtime_mappo_config(args.mappo_config, ROOT)
                mappo_config = load_checkpoint_mappo_config(
                    checkpoint, yaml_config, num_envs=1
                )
                # MAPPOTrainer 初始化时会创建运行目录；统一方法评估只需要最终
                # CSV，因此把中间 trainer 目录放进临时目录，避免污染正式 runs。
                with TemporaryDirectory(prefix="marl_lisl_method_eval_") as temp_dir:
                    eval_mappo = deepcopy(mappo_config)
                    eval_mappo["output"] = dict(eval_mappo["output"])
                    eval_mappo["output"]["run_root"] = Path(temp_dir)
                    eval_mappo["output"]["experiment_name"] = "method_eval"
                    env = LISLMultiFlowEnv(env_config)
                    trainer = MAPPOTrainer(env, eval_mappo, env_config)
                    try:
                        trainer.load_checkpoint(checkpoint, load_optimizer=False)
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
    write_results_csv(results, output_path)


if __name__ == "__main__":
    main()
