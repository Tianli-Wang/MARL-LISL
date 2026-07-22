#!/usr/bin/env python3
"""功能：使用单环境或多进程向量环境训练 MAPPO 策略。"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

def _check_numpy_torch_bridge() -> None:
    """在创建大量环境进程前确认 NumPy 数组能够转换为 PyTorch Tensor。

    Conda 的 ``numpy-base`` 与 pip 的不同版本 ``numpy`` 混装时，普通
    ``import numpy`` 可能成功，但 PyTorch 初始化 NumPy C API 会失败。提前做
    一个最小转换，可以把问题定位在环境依赖，而不是等到 rollout 才报错。
    """
    # 放在函数内部导入非常重要：Windows spawn 会重新执行主脚本的顶层代码。
    # 若在模块顶层导入 torch，所有纯 CPU 环境 worker 都会重复加载 PyTorch，
    # 显著拖慢启动并浪费内存，甚至可能各自触发 CUDA 初始化。
    import numpy as np
    import torch

    try:
        torch.from_numpy(np.zeros(1, dtype=np.float32))
    except (RuntimeError, ImportError) as exc:
        raise SystemExit(
            "PyTorch 无法使用当前 NumPy。请修复同一环境中 pip/Conda NumPy "
            "版本混装后再训练；可先检查 `conda list numpy`，然后统一重装 "
            "NumPy 1.26.4。"
        ) from exc


def main() -> None:
    """创建 LISL 环境和 MAPPOTrainer，启动训练闭环。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", type=Path, default=ROOT / "configs/env.yaml")
    parser.add_argument("--mappo-config", type=Path, default=ROOT / "configs/mappo.yaml")
    parser.add_argument("--num-envs", type=int, default=None, help="临时覆盖并行环境数")
    parser.add_argument("--rollout-length", type=int, default=None, help="临时覆盖每轮采样步数")
    parser.add_argument("--total-updates", type=int, default=None, help="临时覆盖 PPO 更新轮数")
    args = parser.parse_args()
    _check_numpy_torch_bridge()

    # 训练相关重依赖只在父进程进入 main() 后加载；spawn 出来的环境 worker
    # 不会执行该分支，只需导入轻量的环境模块。
    from marl_lisl.algos.mappo import MAPPOTrainer
    from marl_lisl.envs import LISLMultiFlowEnv, SubprocVectorEnv
    from marl_lisl.utils.runtime_config import (
        load_runtime_env_config,
        load_runtime_mappo_config,
    )

    env_config = load_runtime_env_config(args.env_config, ROOT, traffic_split="train")
    mappo_config = load_runtime_mappo_config(args.mappo_config, ROOT)
    # 这些覆盖项只修改本次进程内的配置，适合先用小规模验证完整训练闭环，
    # 不会写回 YAML，也不会影响后续正式实验参数。
    if args.num_envs is not None:
        mappo_config["num_envs"] = max(1, int(args.num_envs))
    if args.rollout_length is not None:
        mappo_config["rollout_length"] = max(1, int(args.rollout_length))
    if args.total_updates is not None:
        mappo_config["total_updates"] = max(1, int(args.total_updates))
    try:
        num_envs = int(mappo_config.get("num_envs", 1))
        if num_envs > 1:
            # 单环境和向量环境满足同一训练接口，但没有共同名义基类。
            env: Any = SubprocVectorEnv(
                env_config,
                num_envs,
                start_method=str(mappo_config.get("multiprocessing_start_method", "auto")),
            )
        else:
            env = LISLMultiFlowEnv(env_config)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\nPrepare data in order: graph snapshots, traffic pairs, then candidates."
        ) from exc
    trainer = MAPPOTrainer(env, mappo_config, env_config)
    try:
        trainer.train()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
