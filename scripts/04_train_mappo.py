#!/usr/bin/env python3
"""功能：训练最小单环境 MAPPO 策略。"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marl_lisl.algos.mappo import MAPPOTrainer
from marl_lisl.envs import LISLMultiFlowEnv
from marl_lisl.utils.config import load_yaml


def _resolve_env_config(path: Path, use_eval: bool = False) -> dict:
    """加载环境配置，并把数据路径解析为项目内绝对路径。"""
    config = load_yaml(path)
    for key in ("graph_dir", "traffic_dir", "traffic_train_path", "traffic_eval_path"):
        config[key] = ROOT / config[key]
    candidates_cfg = dict(config.get("candidates", {}))
    for key in ("train_dir", "eval_dir"):
        if key in candidates_cfg:
            candidates_cfg[key] = ROOT / candidates_cfg[key]
    config["candidates"] = candidates_cfg
    config["future_mutex"] = dict(config["future_mutex"])
    config["future_mutex"]["node_mutex_path"] = ROOT / config["future_mutex"]["node_mutex_path"]
    if use_eval:
        config["traffic_path"] = config["traffic_eval_path"]
        if candidates_cfg.get("enabled", False):
            config["candidate_dir"] = candidates_cfg["eval_dir"]
    elif candidates_cfg.get("enabled", False):
        config["candidate_dir"] = candidates_cfg["train_dir"]
    return config


def _resolve_mappo_config(path: Path) -> dict:
    """加载 MAPPO 配置，并解析实验输出目录。"""
    config = load_yaml(path)
    config["output"] = dict(config["output"])
    config["output"]["run_root"] = ROOT / config["output"]["run_root"]
    return config


def main() -> None:
    """创建 LISL 环境和 MAPPOTrainer，启动训练闭环。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", type=Path, default=ROOT / "configs/env.yaml")
    parser.add_argument("--mappo-config", type=Path, default=ROOT / "configs/mappo.yaml")
    args = parser.parse_args()
    env_config = _resolve_env_config(args.env_config)
    try:
        env = LISLMultiFlowEnv(env_config)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\nPrepare data in order: graph snapshots, traffic pairs, node mutex, then candidates."
        ) from exc
    MAPPOTrainer(env, _resolve_mappo_config(args.mappo_config), env_config).train()


if __name__ == "__main__":
    main()
