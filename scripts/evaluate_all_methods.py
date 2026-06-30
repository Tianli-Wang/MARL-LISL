#!/usr/bin/env python3
"""功能：统一评估 MAPPO、Proactive Rule 与所有 baseline。"""

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marl_lisl.algos.mappo import MAPPOTrainer
from marl_lisl.envs import LISLMultiFlowEnv
from marl_lisl.evaluation import Evaluator
from marl_lisl.evaluation.result_writer import print_diagnostics, print_results_table, write_results_csv
from marl_lisl.evaluation.script_utils import baseline_policies, load_env_config_for_traffic
from marl_lisl.utils.config import load_yaml


class MAPPODeterministicAdapter:
    """Adapter exposing deterministic MAPPO action selection as `act`."""

    def __init__(self, trainer: MAPPOTrainer):
        self.trainer = trainer

    def act(self, obs, state, action_mask):
        actions, *_ = self.trainer.policy.act(obs, state, action_mask, deterministic=True)
        return actions


def _mappo_config(path: Path) -> dict:
    config = load_yaml(path)
    config["num_envs"] = 1
    config["output"] = dict(config["output"])
    config["output"]["run_root"] = ROOT / config["output"]["run_root"]
    return config


def main() -> None:
    """Run all methods and save outputs/tables/all_methods_compare.csv."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", type=Path, default=ROOT / "configs/env.yaml")
    parser.add_argument("--mappo-config", type=Path, default=ROOT / "configs/mappo.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--traffic", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/tables/all_methods_compare.csv")
    parser.add_argument("--preload-graphs", action="store_true", help="评估前预加载全部图，速度更快但占内存")
    args = parser.parse_args()
    env_config = load_env_config_for_traffic(args.env_config, args.traffic, ROOT, args.preload_graphs)
    results = []
    for name, policy in baseline_policies(env_config):
        env = LISLMultiFlowEnv(env_config)
        results.append({"method": name, **Evaluator(env, policy, args.max_steps).run_episode()})

    if args.checkpoint is not None:
        checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
        if not checkpoint.is_file():
            warnings.warn(f"MAPPO checkpoint not found, skipping MAPPO: {checkpoint}", stacklevel=2)
        else:
            try:
                mappo_config = _mappo_config(args.mappo_config)
                env = LISLMultiFlowEnv(env_config)
                trainer = MAPPOTrainer(env, mappo_config, env_config)
                trainer.load_checkpoint(checkpoint)
                policy = MAPPODeterministicAdapter(trainer)
                results.append({"method": "MAPPO", **Evaluator(env, policy, args.max_steps).run_episode()})
            except Exception as exc:
                warnings.warn(f"Skipping MAPPO evaluation: {exc}", stacklevel=2)

    print_results_table(results)
    print_diagnostics(results)
    write_results_csv(results, args.output)


if __name__ == "__main__":
    main()
