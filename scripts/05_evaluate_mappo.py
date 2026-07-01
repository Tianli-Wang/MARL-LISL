#!/usr/bin/env python3
"""功能：使用确定性 masked action 评估 MAPPO checkpoint。"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import sys
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marl_lisl.algos.mappo import MAPPOTrainer
from marl_lisl.envs import LISLMultiFlowEnv
from marl_lisl.utils.runtime_config import (
    load_runtime_env_config,
    load_runtime_mappo_config,
)


def _configs(env_path: Path, mappo_path: Path) -> tuple[dict, dict]:
    """加载评估所需配置，并切换到 evaluation traffic pairs。"""
    env_config = load_runtime_env_config(
        env_path, ROOT, traffic_split="eval", train_random_start=False
    )
    mappo_config = load_runtime_mappo_config(mappo_path, ROOT)
    mappo_config["num_envs"] = 1
    return env_config, mappo_config


def _average_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    """对多个 episode 或 worker 返回的指标做逐字段平均。"""
    return {
        key: float(sum(item[key] for item in items) / max(len(items), 1))
        for key in items[0]
    }


def _evaluate_one_episode(payload: tuple[dict, dict, Path, int]) -> dict[str, float]:
    """子进程入口：独立加载环境和 checkpoint，评估 1 个 episode。"""
    env_config, mappo_config, checkpoint, worker_id = payload
    worker_mappo = deepcopy(mappo_config)
    with TemporaryDirectory(prefix=f"marl_lisl_eval_{worker_id}_") as temp_dir:
        worker_mappo["output"] = dict(worker_mappo["output"])
        worker_mappo["output"]["run_root"] = Path(temp_dir)
        worker_mappo["output"]["experiment_name"] = f"worker_{worker_id}"
        trainer = MAPPOTrainer(LISLMultiFlowEnv(env_config), worker_mappo, env_config)
        trainer.load_checkpoint(checkpoint)
        return trainer.evaluate(1)


def main() -> None:
    """加载 checkpoint，按需并行评估多个 episode，并保存平均指标。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", type=Path, default=ROOT / "configs/env.yaml")
    parser.add_argument("--mappo-config", type=Path, default=ROOT / "configs/mappo.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1, help="并行评估进程数")
    args = parser.parse_args()
    env_config, mappo_config = _configs(args.env_config, args.mappo_config)
    try:
        trainer = MAPPOTrainer(LISLMultiFlowEnv(env_config), mappo_config, env_config)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\nPrepare graph snapshots, evaluation traffic pairs, node mutex, and eval candidates first."
        ) from exc
    trainer.load_checkpoint(args.checkpoint)
    episodes = max(1, int(args.episodes))
    workers = max(1, int(args.workers))
    if workers > 1 and episodes > 1:
        payloads = [
            (env_config, mappo_config, args.checkpoint, index)
            for index in range(episodes)
        ]
        with ProcessPoolExecutor(max_workers=min(workers, episodes)) as executor:
            metrics = _average_metrics(list(executor.map(_evaluate_one_episode, payloads)))
    else:
        metrics = trainer.evaluate(episodes)
    for key, value in metrics.items():
        print(f"{key}={value:.6f}")
    output_path = trainer.metrics_dir / "eval_metrics.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"saved evaluation metrics: {output_path}")


if __name__ == "__main__":
    main()
