#!/usr/bin/env python3
"""功能：统一执行环境基础冒烟测试和 future mutex 专项检查。"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marl_lisl.envs import LISLMultiFlowEnv
from marl_lisl.utils.runtime_config import load_runtime_env_config


def main() -> None:
    """按所选模式 reset 环境、打印互斥详情，并运行合法随机动作。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/env.yaml")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=("train", "eval", "stress"), default="train")
    parser.add_argument(
        "--mode",
        choices=("basic", "mutex", "all"),
        default="all",
        help="basic 只检查 reset/step；mutex 额外打印 future mutex；all 执行全部检查",
    )
    args = parser.parse_args()

    config = load_runtime_env_config(args.config, ROOT, traffic_split=args.split)
    env = LISLMultiFlowEnv(config)
    rng = np.random.default_rng(args.seed)
    obs, state, action_mask = env.reset()
    print(f"obs.shape={obs.shape}, state.shape={state.shape}, mask.shape={action_mask.shape}")

    if args.mode in ("mutex", "all") and env.future_mutex_detector is not None:
        # reset 构造 observation 时已经计算并预热当前路径画像；这里取详细信息
        # 只用于诊断，不会重新执行昂贵的逐边路径可用性检查。
        keep, keep_info = env.future_mutex_detector.compute_future_mutex(
            env.current_paths, env.k
        )
        print(f"future_mutex_keep={keep:.6f}")
        for key in (
            "first_conflict_slot",
            "first_conflict_nodes",
            "invalid_future_path_count",
        ):
            print(f"{key}={keep_info[key]}")
    elif args.mode in ("mutex", "all"):
        print("future_mutex is disabled in the current config")

    for _ in range(max(0, args.steps)):
        actions = np.zeros(env.num_flows, dtype=np.int64)
        for flow_id, mask in enumerate(action_mask):
            legal = np.flatnonzero(mask > 0)
            actions[flow_id] = int(rng.choice(legal)) if len(legal) else 0
        obs, state, action_mask, reward, done, info = env.step(actions)
        print(f"k={info['k']:04d}, actions={actions.tolist()}, reward={reward:.6f}")
        keys = [
            "avg_delay", "peak_delay", "switch_count", "new_link_count",
            "outage_count", "invalid_action_count",
        ]
        if args.mode in ("mutex", "all"):
            keys.append("future_mutex")
        print(", ".join(f"{key}={info[key]}" for key in keys))
        if done:
            break
    print("Environment test finished successfully.")


if __name__ == "__main__":
    main()
