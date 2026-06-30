#!/usr/bin/env python3
"""功能：运行主动规避未来互斥的手工规则策略。"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marl_lisl.envs import LISLMultiFlowEnv, ProactiveRulePolicy
from marl_lisl.utils.config import load_yaml


def _resolved_config(path: Path) -> dict:
    """加载环境配置，并将所有数据文件路径转换为绝对路径。"""
    config = load_yaml(path)
    for key in ("graph_dir", "traffic_dir", "traffic_train_path", "traffic_eval_path"):
        config[key] = ROOT / config[key]
    config["future_mutex"] = dict(config["future_mutex"])
    config["future_mutex"]["node_mutex_path"] = (
        ROOT / config["future_mutex"]["node_mutex_path"]
    )
    return config


def main() -> None:
    """逐步执行 ProactiveRulePolicy，并比较规避前后的 future_mutex。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/env.yaml")
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()
    config = _resolved_config(args.config)
    env = LISLMultiFlowEnv(config)
    rule = config["proactive_rule"]
    policy = ProactiveRulePolicy(
        rule["min_b_avoid"], rule.get("setup_cost_limit"),
        rule.get("prefer_low_delay_when_tie", True),
    )
    obs, _state, action_mask = env.reset()
    limit = env.episode_length if args.steps is None else max(0, args.steps)
    totals = {
        "reward": 0.0, "future_mutex": 0.0, "mutex_avoided": 0.0,
        "switch_count": 0, "new_link_count": 0, "outage_count": 0,
    }
    for _ in range(limit):
        keep, _ = env.future_mutex_detector.compute_future_mutex(env.current_paths, env.k)
        actions = policy.act(obs, action_mask)
        obs, _state, action_mask, reward, done, info = env.step(actions)
        after = float(info["future_mutex"])
        avoided = keep - after
        print(f"k={info['k']:04d}")
        print(f"future_mutex_keep={keep:.6f}")
        print(f"actions={actions.tolist()}")
        print(f"future_mutex_after={after:.6f}")
        print(f"mutex_avoided={avoided:.6f}")
        for key in ("reward", "avg_delay", "switch_count", "new_link_count", "outage_count"):
            print(f"{key}={info[key]}")
        totals["reward"] += reward
        totals["future_mutex"] += after
        totals["mutex_avoided"] += avoided
        for key in ("switch_count", "new_link_count", "outage_count"):
            totals[key] += info[key]
        if done:
            break
    print(f"total_reward={totals['reward']:.6f}")
    print(f"total_future_mutex={totals['future_mutex']:.6f}")
    print(f"total_mutex_avoided={totals['mutex_avoided']:.6f}")
    print(f"total_switch_count={totals['switch_count']}")
    print(f"total_new_link_count={totals['new_link_count']}")
    print(f"total_outage_count={totals['outage_count']}")
    if totals["mutex_avoided"] <= 0:
        print("No mutex was avoided; increase flows, reduce capacity, or adjust traffic pairs.")


if __name__ == "__main__":
    main()
