#!/usr/bin/env python3
"""功能：使用确定性 masked action 评估 MAPPO checkpoint。"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# 默认使用当前已完成训练并通过评估烟测的 latest checkpoint；命令行仍可通过
# --checkpoint 覆盖，便于比较其他 run，而无需修改脚本源码。
DEFAULT_CHECKPOINT = (
    ROOT
    / "outputs/runs/20260701_175147_mappo_a100_16flow_vec/checkpoints/latest.pt"
)

from marl_lisl.utils.runtime_config import (
    load_checkpoint_mappo_config,
    load_runtime_env_config,
    load_runtime_mappo_config,
    resolve_project_path,
)


def _average_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    """对多个独立 episode 返回的同名指标逐字段求平均。"""
    if not items:
        raise ValueError("评估结果不能为空")
    return {
        key: float(sum(item[key] for item in items) / len(items))
        for key in items[0]
    }


def _evaluate_one_episode(
    payload: tuple[dict, dict, Path, int, int | None]
) -> dict[str, float]:
    """子进程入口：独立创建单环境、加载 checkpoint 并确保释放 trainer。"""
    env_config, mappo_config, checkpoint, worker_id, max_steps = payload

    # 重依赖放在 worker 函数内部导入，主进程解析 --help 时不必初始化 CUDA；
    # Windows spawn 子进程则会在这里按需加载一套评估模型。
    from marl_lisl.algos.mappo import MAPPOTrainer
    from marl_lisl.envs import LISLMultiFlowEnv

    worker_mappo = deepcopy(mappo_config)
    with TemporaryDirectory(prefix=f"marl_lisl_eval_{worker_id}_") as temp_dir:
        worker_mappo["output"] = dict(worker_mappo["output"])
        worker_mappo["output"]["run_root"] = Path(temp_dir)
        worker_mappo["output"]["experiment_name"] = f"worker_{worker_id}"
        trainer = MAPPOTrainer(LISLMultiFlowEnv(env_config), worker_mappo, env_config)
        try:
            trainer.load_checkpoint(checkpoint, load_optimizer=False)
            return trainer.evaluate(1, max_steps=max_steps)
        finally:
            trainer.close()


def main() -> None:
    """按需串行或并行评估 checkpoint，并把平均指标保存到指定 JSON。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", type=Path, default=ROOT / "configs/env.yaml")
    parser.add_argument("--mappo-config", type=Path, default=ROOT / "configs/mappo.yaml")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"MAPPO checkpoint，默认：{DEFAULT_CHECKPOINT}",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1, help="并行评估进程数")
    parser.add_argument("--max-steps", type=int, default=None, help="每个 episode 最多评估多少步")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/tables/mappo_eval.json",
        help="评估指标 JSON；相对路径按项目根目录解析",
    )
    args = parser.parse_args()

    checkpoint = resolve_project_path(ROOT, args.checkpoint)
    if not checkpoint.is_file():
        raise SystemExit(f"MAPPO checkpoint 不存在: {checkpoint}")
    output_path = resolve_project_path(ROOT, args.output)
    episodes = max(1, int(args.episodes))
    workers = min(max(1, int(args.workers)), episodes)
    max_steps = None if args.max_steps is None else max(0, int(args.max_steps))

    try:
        env_config = load_runtime_env_config(
            args.env_config, ROOT, traffic_split="eval", train_random_start=False
        )
        yaml_config = load_runtime_mappo_config(args.mappo_config, ROOT)
        # 网络结构优先来自 checkpoint，避免训练后修改 YAML 导致 state_dict
        # 形状不匹配；设备与输出目录仍服从本次评估配置。
        mappo_config = load_checkpoint_mappo_config(
            checkpoint, yaml_config, num_envs=1
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\n请依次准备图快照、evaluation traffic 和 eval candidates。"
        ) from exc

    payloads = [
        (env_config, mappo_config, checkpoint, index, max_steps)
        for index in range(episodes)
    ]
    if workers > 1:
        # Windows 默认 spawn；每个进程独立加载模型。单 GPU 上 worker 太多未必
        # 更快，调用方应按显存和 episode 数控制 --workers。
        with ProcessPoolExecutor(max_workers=workers) as executor:
            items = list(executor.map(_evaluate_one_episode, payloads))
    else:
        items = [_evaluate_one_episode(payload) for payload in payloads]
    metrics = _average_metrics(items)

    for key, value in metrics.items():
        print(f"{key}={value:.6f}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    print(f"saved evaluation metrics: {output_path}")


if __name__ == "__main__":
    main()
