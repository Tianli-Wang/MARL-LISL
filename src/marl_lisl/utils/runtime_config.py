"""运行入口共享的配置加载与项目路径解析工具。"""

from __future__ import annotations

from copy import deepcopy
import warnings
from pathlib import Path

import numpy as np

from marl_lisl.utils.config import load_yaml


_TRAFFIC_KEYS = {
    "train": "traffic_train_path",
    "eval": "traffic_eval_path",
    "stress": "traffic_stress_path",
}
_CANDIDATE_KEYS = {
    "train": "train_dir",
    "eval": "eval_dir",
    "stress": "stress_dir",
}


def resolve_project_path(project_root: Path, path: str | Path) -> Path:
    """把相对路径统一解析到项目根目录，绝对路径则保持不变。"""
    value = Path(path)
    return value if value.is_absolute() else Path(project_root) / value


def load_runtime_env_config(
    config_path: Path,
    project_root: Path,
    *,
    traffic_split: str = "train",
    traffic_path: Path | None = None,
    preload_graphs: bool | None = None,
    train_random_start: bool | None = None,
) -> dict:
    """加载环境配置，并完成所有运行时数据路径和 traffic split 的选择。

    显式传入 ``traffic_path`` 时会优先使用该文件；若它不是配置中登记的
    train/eval/stress 文件，就自动关闭离线候选路径，防止把另一套 traffic 的
    candidate path 错配给当前源宿对。
    """
    project_root = Path(project_root)
    config = load_yaml(config_path)
    for key in (
        "graph_dir",
        "graph_pack_dir",
        "traffic_dir",
        "traffic_train_path",
        "traffic_eval_path",
        "traffic_stress_path",
    ):
        if key in config:
            config[key] = resolve_project_path(project_root, config[key])

    candidates_cfg = dict(config.get("candidates", {}))
    for key in _CANDIDATE_KEYS.values():
        if key in candidates_cfg:
            candidates_cfg[key] = resolve_project_path(project_root, candidates_cfg[key])
    config["candidates"] = candidates_cfg

    # 路径互斥只依赖路径节点集合，不再读取节点容量文件；仍复制子配置，避免
    # 入口脚本临时覆盖 future_window 时修改 YAML 加载器返回的共享对象。
    config["future_mutex"] = dict(config.get("future_mutex", {}))

    if traffic_split not in _TRAFFIC_KEYS:
        raise ValueError(f"未知 traffic split: {traffic_split}")
    selected_traffic = (
        resolve_project_path(project_root, traffic_path)
        if traffic_path is not None
        else Path(config[_TRAFFIC_KEYS[traffic_split]])
    )
    if not selected_traffic.is_file():
        raise FileNotFoundError(f"Traffic 文件不存在: {selected_traffic}")
    config["traffic_path"] = selected_traffic
    # traffic 文件才是实际 agent 数量的直接来源。同步 num_flows 后，stress
    # 场景也能复用同一环境配置，同时训练器仍会检查 MAPPO 维度是否匹配。
    config["num_flows"] = int(np.load(selected_traffic, mmap_mode="r").shape[0])

    matched_split: str | None = None
    for split, config_key in _TRAFFIC_KEYS.items():
        if config_key in config and selected_traffic == Path(config[config_key]):
            matched_split = split
            break
    if candidates_cfg.get("enabled", False):
        candidate_key = _CANDIDATE_KEYS.get(matched_split or "")
        if candidate_key is not None and candidate_key in candidates_cfg:
            config["candidate_dir"] = candidates_cfg[candidate_key]
        else:
            # 自定义 traffic 没有与之对应的离线路径。宁可回退在线生成，也不能
            # 静默读取错误候选，否则动作含义会与源宿对不一致。
            config["candidates"]["enabled"] = False
            warnings.warn(
                "自定义 traffic 没有预计算候选路径，已回退到在线 PathGenerator。",
                stacklevel=2,
            )

    if preload_graphs is not None:
        config["graph_preload"] = bool(preload_graphs)
    config["env"] = dict(config["env"])
    if train_random_start is not None:
        config["env"]["train_random_start"] = bool(train_random_start)
    return config


def load_runtime_mappo_config(config_path: Path, project_root: Path) -> dict:
    """加载 MAPPO 配置，并把实验输出目录转换为绝对路径。"""
    config = load_yaml(config_path)
    config["output"] = dict(config["output"])
    config["output"]["run_root"] = resolve_project_path(
        project_root, config["output"]["run_root"]
    )
    return config


def load_checkpoint_mappo_config(
    checkpoint_path: Path,
    fallback_config: dict,
    *,
    num_envs: int = 1,
) -> dict:
    """优先读取 checkpoint 内保存的 MAPPO 结构，并保留当前运行设置。

    Actor/Critic 的层数、宽度和输入归一化必须与保存权重时一致，因此评估不能
    盲目使用后来修改过的 ``mappo.yaml``。设备和输出目录属于本次运行设置，
    仍采用当前 YAML；旧 checkpoint 没有 config 时才整体回退到当前配置。
    """
    import torch

    checkpoint_path = Path(checkpoint_path)
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # 兼容没有 weights_only 参数的旧 PyTorch
        payload = torch.load(checkpoint_path, map_location="cpu")
    stored_config = payload.get("config") if isinstance(payload, dict) else None
    config = deepcopy(stored_config if isinstance(stored_config, dict) else fallback_config)
    config["device"] = fallback_config.get("device", config.get("device", "cpu"))
    config["output"] = deepcopy(fallback_config["output"])
    config["num_envs"] = max(1, int(num_envs))
    return config
