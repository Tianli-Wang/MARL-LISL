#!/usr/bin/env python3
"""功能：检查未来互斥特征，并用随机合法动作做短轨迹测试。"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marl_lisl.envs import LISLMultiFlowEnv
from marl_lisl.utils.config import load_yaml


def _resolved_config(path: Path) -> dict:
    """加载环境配置，并把图、traffic、mutex 路径解析为绝对路径。"""
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
    """打印保持当前路径的未来互斥量，再执行若干步环境交互。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/env.yaml")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    env = LISLMultiFlowEnv(_resolved_config(args.config))
    rng = np.random.default_rng(args.seed)
    obs, state, action_mask = env.reset()
    print(f"obs.shape={obs.shape}, obs_dim={obs.shape[-1]}")
    print(f"state.shape={state.shape}")
    print(f"action_mask.shape={action_mask.shape}")
    keep, keep_info = env.future_mutex_detector.compute_future_mutex(
        env.current_paths, env.k
    )
    print(f"future_mutex_keep={keep}")
    for key in ("first_conflict_slot", "first_conflict_nodes", "invalid_future_path_count"):
        print(f"{key}={keep_info[key]}")
    if keep == 0:
        print("No future mutex detected. Increase num_flows, reduce node capacity, or adjust traffic pairs.")

    for _ in range(max(0, args.steps)):
        actions = np.zeros(env.num_flows, dtype=np.int64)
        for flow_id, mask in enumerate(action_mask):
            legal = np.flatnonzero(mask > 0)
            actions[flow_id] = int(rng.choice(legal)) if len(legal) else 0
        obs, state, action_mask, reward, done, info = env.step(actions)
        print(
            f"k={info['k']:04d}, reward={reward:.6f}, "
            f"future_mutex={info['future_mutex']:.6f}, "
            f"avg_delay={info['avg_delay']:.6f}, outage_count={info['outage_count']}"
        )
        if done:
            break


if __name__ == "__main__":
    main()
