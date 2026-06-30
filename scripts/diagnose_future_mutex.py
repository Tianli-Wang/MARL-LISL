#!/usr/bin/env python3
"""功能：诊断指定 traffic 场景是否存在 future mutex 压力。"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marl_lisl.baselines import MaintainUntilConflictPolicy, ProactiveRuleBaseline, ShortestDelayPolicy
from marl_lisl.envs import LISLMultiFlowEnv
from marl_lisl.evaluation import Evaluator
from marl_lisl.evaluation.result_writer import print_diagnostics, print_results_table
from marl_lisl.evaluation.script_utils import load_env_config_for_traffic


def main() -> None:
    """Run a small set of policies and print mutex-pressure diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/env.yaml")
    parser.add_argument("--traffic", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--preload-graphs", action="store_true", help="诊断前预加载全部图，速度更快但占内存")
    args = parser.parse_args()
    config = load_env_config_for_traffic(args.config, args.traffic, ROOT, args.preload_graphs)
    policies = [
        ("MaintainUntilConflict", MaintainUntilConflictPolicy()),
        ("ShortestDelay", ShortestDelayPolicy()),
        ("ProactiveRule", ProactiveRuleBaseline.from_config(config)),
    ]
    results = []
    for name, policy in policies:
        env = LISLMultiFlowEnv(config)
        results.append({"method": name, **Evaluator(env, policy, args.max_steps).run_episode()})
    print_results_table(results)
    print_diagnostics(results)


if __name__ == "__main__":
    main()
