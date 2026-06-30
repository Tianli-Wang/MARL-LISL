#!/usr/bin/env python3
"""功能：统一运行 Shortest/Maintain/Greedy/Proactive baselines 并输出对比表。"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marl_lisl.envs import LISLMultiFlowEnv
from marl_lisl.evaluation import Evaluator
from marl_lisl.evaluation.result_writer import print_diagnostics, print_results_table, write_results_csv
from marl_lisl.evaluation.script_utils import baseline_policies, load_env_config_for_traffic


def main() -> None:
    """Run all baselines and save outputs/tables/baseline_compare.csv."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/env.yaml")
    parser.add_argument("--traffic", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/tables/baseline_compare.csv")
    parser.add_argument("--preload-graphs", action="store_true", help="评估前预加载全部图，速度更快但占内存")
    args = parser.parse_args()
    config = load_env_config_for_traffic(args.config, args.traffic, ROOT, args.preload_graphs)
    results = []
    for name, policy in baseline_policies(config):
        env = LISLMultiFlowEnv(config)
        row = {"method": name, **Evaluator(env, policy, args.max_steps).run_episode()}
        results.append(row)
    print_results_table(results)
    print_diagnostics(results)
    write_results_csv(results, args.output)



if __name__ == "__main__":
    main()
