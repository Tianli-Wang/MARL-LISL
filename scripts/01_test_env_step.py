#!/usr/bin/env python3
"""功能：用合法随机动作快速检查 LISL 环境 reset/step 是否可运行。"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marl_lisl.envs import LISLMultiFlowEnv
from marl_lisl.utils.config import load_yaml


def _resolve_data_paths(config: dict) -> dict:
    """把配置中的相对数据路径转换为项目根目录下的绝对路径。"""
    config = dict(config)
    for key in ("graph_dir", "traffic_dir", "traffic_train_path", "traffic_eval_path"):
        config[key] = ROOT / config[key]
    candidates_cfg = dict(config.get("candidates", {}))
    for key in ("train_dir", "eval_dir"):
        if key in candidates_cfg:
            candidates_cfg[key] = ROOT / candidates_cfg[key]
    config["candidates"] = candidates_cfg
    mutex_cfg = dict(config.get("future_mutex", {}))
    if "node_mutex_path" in mutex_cfg:
        mutex_cfg["node_mutex_path"] = ROOT / mutex_cfg["node_mutex_path"]
    config["future_mutex"] = mutex_cfg
    return config


def main() -> None:
    """构造环境，采样合法随机动作，并打印每步关键统计。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/env.yaml")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval", action="store_true", help="使用评估 traffic pairs")
    args = parser.parse_args()

    config = _resolve_data_paths(load_yaml(args.config))
    if args.eval:
        config["traffic_path"] = config["traffic_eval_path"]
        if config.get("candidates", {}).get("enabled", False):
            config["candidate_dir"] = config["candidates"]["eval_dir"]
    elif config.get("candidates", {}).get("enabled", False):
        config["candidate_dir"] = config["candidates"]["train_dir"]
    rng = np.random.default_rng(args.seed)
    env = LISLMultiFlowEnv(config)
    obs, state, action_mask = env.reset()
    print(f"obs.shape={obs.shape}, state.shape={state.shape}, mask.shape={action_mask.shape}")

    for _ in range(max(0, args.steps)):
        actions = np.zeros(env.num_flows, dtype=np.int64)
        for flow_id in range(env.num_flows):
            legal = np.flatnonzero(action_mask[flow_id] > 0)
            actions[flow_id] = int(rng.choice(legal)) if len(legal) else 0
        obs, state, action_mask, reward, done, info = env.step(actions)
        print(f"k={info['k']:04d}")
        print(f"actions={actions.tolist()}")
        print(f"reward={reward:.6f}")
        for key in (
            "avg_delay", "peak_delay", "switch_count", "new_link_count",
            "outage_count", "invalid_action_count",
        ):
            print(f"{key}={info[key]}")
        if done:
            break
    print("Environment step test finished successfully.")


if __name__ == "__main__":
    main()
