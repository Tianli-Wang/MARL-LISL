#!/usr/bin/env python3
"""功能：运行主动规避未来互斥的手工规则策略。"""

import argparse
import sys
import time
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
    candidates_cfg = dict(config.get("candidates", {}))
    for key in ("train_dir", "eval_dir"):
        if key in candidates_cfg:
            candidates_cfg[key] = ROOT / candidates_cfg[key]
    config["candidates"] = candidates_cfg
    if candidates_cfg.get("enabled", False):
        config["candidate_dir"] = candidates_cfg["train_dir"]
    config["future_mutex"] = dict(config["future_mutex"])
    config["future_mutex"]["node_mutex_path"] = (
        ROOT / config["future_mutex"]["node_mutex_path"]
    )
    return config


def main() -> None:
    """逐步执行 ProactiveRulePolicy，并比较规避前后的 future_mutex。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/env.yaml")
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="最多运行多少步；默认只跑 20 步用于快速验证",
    )
    parser.add_argument(
        "--full-episode",
        action="store_true",
        help="显式跑完整 episode，会按 env.episode_length 执行，可能很慢",
    )
    parser.add_argument("--future-window", type=int, default=None, help="临时覆盖 future_mutex.future_window")
    parser.add_argument("--num-candidates", type=int, default=None, help="临时覆盖每条流的候选路径数")
    parser.add_argument("--parallel-workers", type=int, default=None, help="临时覆盖环境候选路径并行线程数")
    args = parser.parse_args()
    config = _resolved_config(args.config)
    if args.future_window is not None:
        config["future_mutex"]["future_window"] = max(0, int(args.future_window))
    if args.num_candidates is not None:
        config["num_candidates"] = max(1, int(args.num_candidates))
    if args.parallel_workers is not None:
        config["parallel_workers"] = max(1, int(args.parallel_workers))
    env = LISLMultiFlowEnv(config)
    rule = config["proactive_rule"]
    policy = ProactiveRulePolicy(
        rule["min_b_avoid"], rule.get("setup_cost_limit"),
        rule.get("prefer_low_delay_when_tie", True),
    )
    reset_start = time.perf_counter()
    obs, _state, action_mask = env.reset()
    print(f"reset_time_s={time.perf_counter() - reset_start:.3f}")
    limit = env.episode_length if args.full_episode else max(0, args.steps)
    print(f"running proactive rule for at most {limit} steps")
    totals = {
        "reward": 0.0, "future_mutex": 0.0, "mutex_avoided": 0.0,
        "switch_count": 0, "new_link_count": 0, "outage_count": 0,
    }
    for _ in range(limit):
        step_start = time.perf_counter()
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
        print(f"step_time_s={time.perf_counter() - step_start:.3f}")
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
