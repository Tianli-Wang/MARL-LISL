#!/usr/bin/env python3
"""功能：使用单环境或多进程向量环境训练 MAPPO 策略。"""

import argparse
import os
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marl_lisl.algos.mappo import MAPPOTrainer
from marl_lisl.envs import LISLMultiFlowEnv, SubprocVectorEnv
from marl_lisl.utils.runtime_config import (
    load_runtime_env_config,
    load_runtime_mappo_config,
)


def main() -> None:
    """创建 LISL 环境和 MAPPOTrainer，启动训练闭环。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", type=Path, default=ROOT / "configs/env.yaml")
    parser.add_argument("--mappo-config", type=Path, default=ROOT / "configs/mappo.yaml")
    args = parser.parse_args()
    env_config = load_runtime_env_config(args.env_config, ROOT, traffic_split="train")
    mappo_config = load_runtime_mappo_config(args.mappo_config, ROOT)
    try:
        num_envs = int(mappo_config.get("num_envs", 1))
        if num_envs > 1:
            env = SubprocVectorEnv(
                env_config,
                num_envs,
                start_method=str(mappo_config.get("multiprocessing_start_method", "spawn")),
            )
        else:
            env = LISLMultiFlowEnv(env_config)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\nPrepare data in order: graph snapshots, traffic pairs, node mutex, then candidates."
        ) from exc
    trainer = MAPPOTrainer(env, mappo_config, env_config)
    try:
        trainer.train()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
